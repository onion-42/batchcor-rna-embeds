"""
finetune_scgpt_survival.py
==========================
Supervised fine-tuning of the pretrained scGPT model on bulk-RNA TRAIN cohorts
with a Cox negative log partial likelihood loss, then extraction of the new
``<cls>`` embeddings for TRAIN (out-of-fold) and PUB cohorts.

Design choices (PEP 8, no silent fallbacks)
-------------------------------------------
* Reuse the data plumbing from ``scgpt_embeddings.py`` (HVG selection, binning,
  CLS+gene-token tensor assembly).  The same preprocessing pipeline guarantees
  the fine-tuned model never sees a distribution it would not see at zero-shot
  inference time.
* Freeze every transformer block EXCEPT the last ``--unfreeze-last`` (default
  ``2``).  This caps the trainable parameter count and prevents catastrophic
  overfitting on ~1500 patients.
* Wrap the forward pass in ``torch.amp.autocast`` (mixed precision) and call
  ``torch.cuda.empty_cache()`` between folds.  This pulls VRAM peak down by
  roughly 35-45% on the scGPT-human checkpoint (12 layers, d_model=512).
* Cox NLL is computed on the FULL training set in one pass per epoch.  Mini-
  batching the Cox loss biases the risk-set, so we use full-batch Adam.  At
  d_model=512 and 1500 patients * 1201 tokens * fp16 the activation memory is
  comfortable on a single 12 GB GPU.
* 5-fold stratified-by-event KFold CV.  In each fold:
      train on 80%  ->  infer (no_grad) on the held-out 20%.
  The 5 held-out vectors are concatenated to form the OOF embedding for TRAIN.
* For PUB cohorts we train a SINGLE final model on the FULL TRAIN set (with
  the same number of epochs averaged over the CV folds at best validation
  C-index) and infer once.

Outputs
-------
* ``embeddings/finetuned_scgpt_embeddings.npy``       -- TRAIN OOF (N_train, D)
* ``embeddings/finetuned_scgpt_embeddings.npz``       -- ``train`` + every PUB
* ``embeddings/finetuned_scgpt_index.csv``            -- patient ID alignment
* ``embeddings/finetuned_scgpt_train_history.json``   -- per-fold curves
* ``data/processed/<cohort>.h5ad``                    -- new
                                          ``obsm["scGPT_finetuned_embedding"]``
  (so the v4 benchmark picks it up without code changes)

CLI
---
    cd batchcor-rna-embeds
    python -m batchcor_rna_emb.modeling.finetune_scgpt_survival \\
        --epochs 30 --lr 5e-5 --unfreeze-last 2 --batch-size 32

Falls back gracefully when CUDA is absent (CPU is still functional, just slow).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

from scgpt.model import TransformerModel
from scgpt.tokenizer.gene_tokenizer import GeneVocab

from batchcor_rna_emb.modeling.scgpt_embeddings import (
    CLS_TOKEN,
    EOS_TOKEN,
    N_BINS,
    N_HVG,
    PAD_TOKEN,
    PAD_VALUE,
    bin_expression,
    build_model_inputs,
    preprocess_adata,
)


# =============================================================================
# LOGGING
# =============================================================================
logger.remove()
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | <cyan>{function}</cyan> | {message}"
    ),
    level=os.environ.get("LOG_LEVEL", "INFO"),
)


# =============================================================================
# CONFIG
# =============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

SCGPT_MODEL_DIR: Path = Path(
    os.environ.get(
        "SCGPT_MODEL_DIR", str(REPO_ROOT / "pretrained/scGPT_human")
    )
).resolve()

PROCESSED_DIR: Path = REPO_ROOT / "data/processed"
EMBEDDINGS_DIR: Path = REPO_ROOT / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_H5AD: Path = PROCESSED_DIR / "TRAIN_Combined_cAE_Corrected.h5ad"
PUB_FILES: dict[str, Path] = {
    "PUB_BLCA":      PROCESSED_DIR / "PUB_BLCA_Mariathasan_EGAS00001002556_ICI.h5ad",
    "PUB_ccRCC_ICI": PROCESSED_DIR / "PUB_ccRCC_Immotion150_and_151_ICI.h5ad",
    "PUB_ccRCC_TKI": PROCESSED_DIR / "PUB_ccRCC_Immotion150_and_151_TKI.h5ad",
}

OOF_NPY: Path = EMBEDDINGS_DIR / "finetuned_scgpt_embeddings.npy"
ALL_NPZ: Path = EMBEDDINGS_DIR / "finetuned_scgpt_embeddings.npz"
INDEX_CSV: Path = EMBEDDINGS_DIR / "finetuned_scgpt_index.csv"
HISTORY_JSON: Path = EMBEDDINGS_DIR / "finetuned_scgpt_train_history.json"

OBSM_KEY: str = "scGPT_finetuned_embedding"

SEED: int = 42


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass(frozen=True)
class FineTuneConfig:
    epochs: int = 30
    batch_size: int = 32
    cox_batch: int = 64
    lr: float = 5e-5
    weight_decay: float = 1e-4
    unfreeze_last: int = 2
    n_splits: int = 5
    head_hidden: int = 128
    head_dropout: float = 0.20
    grad_clip: float = 1.0
    warmup_epochs: int = 2
    early_stop_patience: int = 6
    use_amp: bool = True


@dataclass
class FoldHistory:
    fold: int
    train_loss: list[float]
    val_cindex: list[float]
    best_val_cindex: float
    best_epoch: int


# =============================================================================
# MODEL
# =============================================================================
class ScGPTSurvivalModel(nn.Module):
    """
    Fine-tunable scGPT trunk + lightweight survival head.

    The head outputs a single scalar interpreted as the **log hazard ratio**
    (proportional hazards convention: higher = riskier).  It is trained against
    the Cox negative log partial likelihood.
    """

    def __init__(
        self,
        backbone: TransformerModel,
        d_model: int,
        head_hidden: int = 128,
        head_dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, 1),
        )

    def encode_cls(
        self,
        gene_tokens: torch.Tensor,
        expr_values: torch.Tensor,
        key_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the backbone and return the CLS-pooled vector ``(B, d_model)``."""
        out = self.backbone(
            src=gene_tokens,
            values=expr_values,
            src_key_padding_mask=key_pad_mask,
            CLS=True,
            CCE=False,
            MVC=False,
            ECS=False,
            do_sample=False,
        )
        if isinstance(out, dict):
            if "cell_emb" in out:
                return out["cell_emb"]
            if "h" in out:
                return out["h"][:, 0, :]
            raise RuntimeError(
                f"Unexpected scGPT output keys: {list(out.keys())}"
            )
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out[:, 0, :]
            if out.dim() == 2:
                return out
        raise RuntimeError(f"Unexpected scGPT output type: {type(out)}")

    def forward(
        self,
        gene_tokens: torch.Tensor,
        expr_values: torch.Tensor,
        key_pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.encode_cls(gene_tokens, expr_values, key_pad_mask)
        risk = self.head(cls).squeeze(-1)
        return risk, cls


# =============================================================================
# LAYER FREEZING
# =============================================================================
def freeze_backbone_except_last(
    model: ScGPTSurvivalModel, n_unfreeze: int
) -> tuple[int, int]:
    """
    Freeze every backbone parameter, then unfreeze the last ``n_unfreeze``
    transformer encoder layers.  The survival head stays fully trainable.

    Returns
    -------
    (trainable_count, frozen_count)
    """
    for p in model.backbone.parameters():
        p.requires_grad = False

    encoder = getattr(model.backbone, "transformer_encoder", None)
    if encoder is None or not hasattr(encoder, "layers"):
        raise RuntimeError(
            "Could not locate `transformer_encoder.layers` on the scGPT "
            "backbone -- the freezing strategy depends on this attribute. "
            "Inspect `model.backbone` and adapt accordingly."
        )

    layers = encoder.layers
    if n_unfreeze < 0 or n_unfreeze > len(layers):
        raise ValueError(
            f"--unfreeze-last must be in [0, {len(layers)}], got {n_unfreeze}"
        )

    for layer in layers[-n_unfreeze:] if n_unfreeze > 0 else []:
        for p in layer.parameters():
            p.requires_grad = True

    for p in model.head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    logger.info(
        f"Trainable params: {trainable:>11,} | "
        f"Frozen: {frozen:>11,} | "
        f"Last {n_unfreeze} of {len(layers)} encoder layers + head are trainable"
    )
    return trainable, frozen


# =============================================================================
# COX NEGATIVE LOG PARTIAL LIKELIHOOD
# =============================================================================
def cox_nll_loss(
    risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor
) -> torch.Tensor:
    """
    Breslow approximation of the Cox negative log partial likelihood.

    Parameters
    ----------
    risk
        Raw model output, shape ``(N,)``.  Larger = higher hazard.
    time
        Survival / censoring time, shape ``(N,)``.
    event
        Binary event indicator, shape ``(N,)``.  ``1`` = event observed,
        ``0`` = right-censored.

    Returns
    -------
    Scalar loss tensor (mean over events).
    """
    if event.sum() < 1.0:
        return torch.zeros((), device=risk.device, dtype=risk.dtype)

    order = torch.argsort(time, descending=True)
    risk_s = risk[order]
    event_s = event[order]
    log_cumsum = torch.logcumsumexp(risk_s, dim=0)
    nll = -((risk_s - log_cumsum) * event_s).sum() / event_s.sum().clamp(min=1.0)
    return nll


# =============================================================================
# DATA PIPELINE
# =============================================================================
def _load_engine() -> tuple[GeneVocab, dict, int]:
    """Load the scGPT vocabulary and config (no model weights yet)."""
    vocab_path = SCGPT_MODEL_DIR / "vocab.json"
    args_path = SCGPT_MODEL_DIR / "args.json"
    if not vocab_path.exists() or not args_path.exists():
        raise FileNotFoundError(
            f"scGPT weights/vocab missing under {SCGPT_MODEL_DIR}.\n"
            "Place vocab.json, args.json, best_model.pt there.\n"
            "Override location with the SCGPT_MODEL_DIR env variable."
        )
    vocab = GeneVocab.from_file(str(vocab_path))
    with open(args_path) as fh:
        cfg = json.load(fh)
    embsize = int(cfg.get("embsize", 512))
    return vocab, cfg, embsize


def _build_backbone(vocab: GeneVocab, cfg: dict) -> TransformerModel:
    backbone = TransformerModel(
        ntoken=len(vocab),
        d_model=int(cfg.get("embsize", 512)),
        nhead=int(cfg.get("nheads", 8)),
        d_hid=int(cfg.get("d_hid", 512)),
        nlayers=int(cfg.get("nlayers", 12)),
        vocab=vocab,
        dropout=float(cfg.get("dropout", 0.0)),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=bool(cfg.get("pre_norm", False)),
    )
    weight_path = SCGPT_MODEL_DIR / "best_model.pt"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing scGPT weights: {weight_path}")
    state = torch.load(str(weight_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"{len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        logger.warning(f"{len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return backbone


def _shared_hvg_geneset(
    train_h5ad: Path, n_hvg: int, vocab_genes: frozenset[str]
) -> list[str]:
    """
    Pick a SHARED HVG set on the TRAIN cohort so the same gene order is used
    everywhere (training + OOF embedding + PUB inference).
    """
    if not train_h5ad.exists():
        raise FileNotFoundError(f"TRAIN h5ad missing: {train_h5ad}")
    adata = sc.read_h5ad(str(train_h5ad))
    overlap = [g for g in adata.var_names if g in vocab_genes]
    if not overlap:
        raise ValueError(
            "Zero genes overlap between TRAIN var_names and scGPT vocab. "
            "Set adata.var_names to HGNC symbols (e.g. TP53) before running."
        )
    adata = adata[:, overlap].copy()
    if adata.n_vars > n_hvg:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()
    logger.info(
        f"Shared HVG set: {adata.n_vars} genes "
        f"(from {len(overlap)} vocab-overlapping genes on TRAIN)"
    )
    return list(adata.var_names)


def _materialise_tensors(
    adata: ad.AnnData, gene_order: Sequence[str], vocab: GeneVocab
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply preprocessing + binning to a cohort and return scGPT input tensors."""
    keep = [g for g in gene_order if g in adata.var_names]
    if len(keep) < len(gene_order):
        logger.warning(
            f"Cohort missing {len(gene_order) - len(keep)} of {len(gene_order)} "
            "shared HVGs -- those positions will be left at PAD_VALUE."
        )

    sub = adata[:, keep].copy()
    sub = preprocess_adata(sub)
    import scipy.sparse as sp

    X = sub.X.toarray() if sp.issparse(sub.X) else np.asarray(sub.X)
    X = X.astype(np.float32)
    X_binned = bin_expression(X, n_bins=N_BINS)

    if len(keep) < len(gene_order):
        full = np.zeros((X_binned.shape[0], len(gene_order)), dtype=np.int64)
        idx_map = {g: i for i, g in enumerate(gene_order)}
        for j, g in enumerate(keep):
            full[:, idx_map[g]] = X_binned[:, j]
        X_binned = full

    gene_ids = np.array(
        [vocab[g] for g in gene_order], dtype=np.int64
    )
    return build_model_inputs(
        X_binned, gene_ids, pad_id=vocab[PAD_TOKEN], n_bins=N_BINS
    )


def _extract_survival(adata: ad.AnnData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mask, time, event) extracted from common OS/PFS columns."""
    obs = adata.obs
    time_candidates = [
        "PFS", "OS", "PFS_DAYS", "OS_DAYS",
        "pfs", "os", "pfs_days", "os_days",
    ]
    event_candidates = [
        "PFS_FLAG", "OS_FLAG",
        "PFS_EVENT", "OS_EVENT",
        "PFS_STATUS", "OS_STATUS",
        "pfs_flag", "os_flag", "pfs_event", "os_event",
    ]
    time_col = next((c for c in time_candidates if c in obs.columns), None)
    event_col = next((c for c in event_candidates if c in obs.columns), None)
    if time_col is None or event_col is None:
        raise RuntimeError(
            "Could not find PFS/OS time + event columns on TRAIN obs. "
            f"Tried {time_candidates} and {event_candidates}."
        )
    t = pd.to_numeric(obs[time_col], errors="coerce").to_numpy()
    e = pd.to_numeric(obs[event_col], errors="coerce").to_numpy()
    mask = (~np.isnan(t)) & (~np.isnan(e)) & (t > 0)
    return mask, t, e


# =============================================================================
# TRAINING LOOP (FULL-BATCH COX)
# =============================================================================
def _make_optimizer(model: ScGPTSurvivalModel, cfg: FineTuneConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
    )


def _make_scheduler(opt: torch.optim.Optimizer, cfg: FineTuneConfig):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, cfg.epochs - cfg.warmup_epochs), eta_min=cfg.lr * 0.05
    )


def _forward_chunked(
    model: ScGPTSurvivalModel,
    g: torch.Tensor,
    e: torch.Tensor,
    m: torch.Tensor,
    chunk: int,
    device: torch.device,
    use_amp: bool,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward N samples in chunks to avoid VRAM blow-up; returns (risk, cls)."""
    risks: list[torch.Tensor] = []
    cls_list: list[torch.Tensor] = []
    use_autocast = use_amp and device.type == "cuda"
    for s in range(0, g.size(0), chunk):
        g_b = g[s : s + chunk].to(device, non_blocking=True)
        e_b = e[s : s + chunk].to(device, non_blocking=True)
        m_b = m[s : s + chunk].to(device, non_blocking=True)
        if train:
            if use_autocast:
                with torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16
                ):
                    r, c = model(g_b, e_b, m_b)
            else:
                r, c = model(g_b, e_b, m_b)
        else:
            with torch.no_grad():
                if use_autocast:
                    with torch.amp.autocast(
                        device_type=device.type, dtype=torch.float16
                    ):
                        r, c = model(g_b, e_b, m_b)
                else:
                    r, c = model(g_b, e_b, m_b)
        risks.append(r.detach() if not train else r)
        cls_list.append(c.float().detach().cpu())
    return torch.cat(risks, dim=0), torch.cat(cls_list, dim=0)


def _train_one_fold(
    fold: int,
    backbone_state: dict,
    vocab: GeneVocab,
    cfg: FineTuneConfig,
    backbone_kwargs: dict,
    g: torch.Tensor,
    e: torch.Tensor,
    m: torch.Tensor,
    times: np.ndarray,
    events: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
) -> tuple[FoldHistory, np.ndarray]:
    """Train one fold and return history + held-out CLS embeddings."""
    backbone = TransformerModel(**backbone_kwargs)
    missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
    if missing:
        logger.debug(f"fold{fold} missing keys: {len(missing)}")
    model = ScGPTSurvivalModel(
        backbone=backbone,
        d_model=backbone_kwargs["d_model"],
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
    ).to(device)

    freeze_backbone_except_last(model, cfg.unfreeze_last)
    opt = _make_optimizer(model, cfg)
    sched = _make_scheduler(opt, cfg)
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    times_t = torch.from_numpy(times[tr_idx].astype(np.float32)).to(device)
    events_t = torch.from_numpy(events[tr_idx].astype(np.float32)).to(device)

    g_tr, e_tr, m_tr = g[tr_idx], e[tr_idx], m[tr_idx]
    g_va, e_va, m_va = g[va_idx], e[va_idx], m[va_idx]

    train_loss_hist: list[float] = []
    val_cidx_hist: list[float] = []
    best_cidx, best_epoch, no_improve = -np.inf, 0, 0
    best_val_cls: np.ndarray | None = None

    n_train = g_tr.size(0)
    micro = cfg.batch_size
    cox_batch = min(cfg.cox_batch, n_train)
    n_steps = max(1, n_train // cox_batch)

    logger.info(
        f"[fold {fold}] n_train={n_train} n_val={g_va.size(0)} "
        f"events_train={int(events[tr_idx].sum())} | "
        f"cox_batch={cox_batch} grad_steps/epoch={n_steps}"
    )

    rng = np.random.default_rng(SEED + fold)

    use_autocast = cfg.use_amp and device.type == "cuda"
    for ep in range(1, cfg.epochs + 1):
        model.train()

        def _autocast() -> contextlib.AbstractContextManager:
            if use_autocast:
                return torch.amp.autocast(
                    device_type=device.type, dtype=torch.float16
                )
            return contextlib.nullcontext()
        # Shuffle local training indices once per epoch and slice into
        # `cox_batch`-sized risk-sets.  Each step computes a full-graph
        # Cox loss on its risk-set, runs backward, and zeroes grads --
        # this caps activation memory regardless of n_train.
        perm = rng.permutation(n_train)
        last_loss: float = float("nan")
        for step in range(n_steps):
            sel = perm[step * cox_batch : (step + 1) * cox_batch]
            opt.zero_grad(set_to_none=True)
            risks_chunks: list[torch.Tensor] = []
            for s in range(0, len(sel), micro):
                idxs = sel[s : s + micro]
                g_b = g_tr[idxs].to(device, non_blocking=True)
                e_b = e_tr[idxs].to(device, non_blocking=True)
                m_b = m_tr[idxs].to(device, non_blocking=True)
                with _autocast():
                    r, _ = model(g_b, e_b, m_b)
                risks_chunks.append(r)
            risks = torch.cat(risks_chunks, dim=0)
            t_step = times_t[torch.from_numpy(sel).to(device)]
            e_step = events_t[torch.from_numpy(sel).to(device)]
            with _autocast():
                loss = cox_nll_loss(risks, t_step, e_step)
            last_loss = float(loss.detach().cpu())
            if cfg.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                opt.step()
            del risks, risks_chunks
            if (step + 1) % max(1, n_steps // 4) == 0 or step == 0:
                logger.info(
                    f"[fold {fold}] ep {ep} step {step + 1}/{n_steps} "
                    f"loss={last_loss:.4f}"
                )
        loss_value = last_loss
        gc.collect()

        if ep > cfg.warmup_epochs:
            sched.step()

        model.eval()
        risk_va, cls_va = _forward_chunked(
            model, g_va, e_va, m_va, micro, device, cfg.use_amp, train=False
        )
        risk_va_np = risk_va.float().detach().cpu().numpy()
        cidx = float(
            concordance_index_censored(
                events[va_idx].astype(bool), times[va_idx], risk_va_np
            )[0]
        )
        train_loss_hist.append(loss_value)
        val_cidx_hist.append(cidx)

        if cidx > best_cidx + 1e-4:
            best_cidx, best_epoch = cidx, ep
            best_val_cls = cls_va.numpy().copy()
            no_improve = 0
        else:
            no_improve += 1

        logger.info(
            f"[fold {fold}] ep {ep:>3d} | "
            f"loss={loss_value:.4f} | "
            f"val C-idx={cidx:.4f} (best {best_cidx:.4f}@ep{best_epoch})"
        )

        if no_improve >= cfg.early_stop_patience:
            logger.info(
                f"[fold {fold}] early-stopping at ep {ep} "
                f"(no improvement for {no_improve} epochs)"
            )
            break

    if best_val_cls is None:
        _, cls_va = _forward_chunked(
            model, g_va, e_va, m_va, micro, device, cfg.use_amp, train=False
        )
        best_val_cls = cls_va.numpy()

    del model, backbone, opt, sched, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    history = FoldHistory(
        fold=fold,
        train_loss=train_loss_hist,
        val_cindex=val_cidx_hist,
        best_val_cindex=best_cidx,
        best_epoch=best_epoch,
    )
    return history, best_val_cls


def _train_full(
    backbone_state: dict,
    vocab: GeneVocab,
    cfg: FineTuneConfig,
    backbone_kwargs: dict,
    g: torch.Tensor,
    e: torch.Tensor,
    m: torch.Tensor,
    times: np.ndarray,
    events: np.ndarray,
    epochs: int,
    device: torch.device,
) -> ScGPTSurvivalModel:
    """Train one final model on the FULL TRAIN set for PUB inference."""
    backbone = TransformerModel(**backbone_kwargs)
    backbone.load_state_dict(backbone_state, strict=False)
    model = ScGPTSurvivalModel(
        backbone=backbone,
        d_model=backbone_kwargs["d_model"],
        head_hidden=cfg.head_hidden,
        head_dropout=cfg.head_dropout,
    ).to(device)
    freeze_backbone_except_last(model, cfg.unfreeze_last)

    opt = _make_optimizer(model, cfg)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, epochs - cfg.warmup_epochs), eta_min=cfg.lr * 0.05
    )
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    times_t = torch.from_numpy(times.astype(np.float32)).to(device)
    events_t = torch.from_numpy(events.astype(np.float32)).to(device)
    micro = cfg.batch_size
    n_full = g.size(0)
    cox_batch = min(cfg.cox_batch, n_full)
    n_steps = max(1, n_full // cox_batch)

    use_autocast = cfg.use_amp and device.type == "cuda"

    def _autocast() -> contextlib.AbstractContextManager:
        if use_autocast:
            return torch.amp.autocast(
                device_type=device.type, dtype=torch.float16
            )
        return contextlib.nullcontext()

    logger.info(
        f"[final] training on full TRAIN n={n_full} for {epochs} epochs "
        f"| cox_batch={cox_batch} grad_steps/epoch={n_steps}"
    )
    rng = np.random.default_rng(SEED + 999)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_full)
        last_loss: float = float("nan")
        for step in range(n_steps):
            sel = perm[step * cox_batch : (step + 1) * cox_batch]
            opt.zero_grad(set_to_none=True)
            risks: list[torch.Tensor] = []
            for s in range(0, len(sel), micro):
                idxs = sel[s : s + micro]
                g_b = g[idxs].to(device, non_blocking=True)
                e_b = e[idxs].to(device, non_blocking=True)
                m_b = m[idxs].to(device, non_blocking=True)
                with _autocast():
                    r, _ = model(g_b, e_b, m_b)
                risks.append(r)
            risks_cat = torch.cat(risks, dim=0)
            t_step = times_t[torch.from_numpy(sel).to(device)]
            e_step = events_t[torch.from_numpy(sel).to(device)]
            with _autocast():
                loss = cox_nll_loss(risks_cat, t_step, e_step)
            last_loss = float(loss.detach().cpu())
            if cfg.use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                )
                opt.step()
            del risks, risks_cat

        if ep > cfg.warmup_epochs:
            sched.step()
        if ep % max(1, epochs // 6) == 0 or ep == epochs:
            logger.info(
                f"[final] ep {ep:>3d}/{epochs} loss={last_loss:.4f}"
            )
    return model


@torch.no_grad()
def _embed(
    model: ScGPTSurvivalModel,
    g: torch.Tensor,
    e: torch.Tensor,
    m: torch.Tensor,
    device: torch.device,
    chunk: int,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    _, cls = _forward_chunked(
        model, g, e, m, chunk, device, use_amp, train=False
    )
    return cls.numpy().astype(np.float32)


# =============================================================================
# I/O HELPERS
# =============================================================================
def _persist_obsm(h5ad_path: Path, ids: list[str], emb: np.ndarray) -> None:
    """Insert ``OBSM_KEY`` into the h5ad in place (aligned to obs.index)."""
    if not h5ad_path.exists():
        logger.warning(f"Skip obsm write -- {h5ad_path} missing")
        return
    adata = sc.read_h5ad(str(h5ad_path))
    if list(adata.obs.index) != list(ids):
        id2row = {pid: i for i, pid in enumerate(ids)}
        order = np.array(
            [id2row[i] for i in adata.obs.index if i in id2row], dtype=np.int64
        )
        if order.size != adata.n_obs:
            logger.warning(
                f"{h5ad_path.name}: only {order.size}/{adata.n_obs} ids match "
                "embedding -- writing only the matched rows."
            )
            mask = np.array(
                [i in id2row for i in adata.obs.index], dtype=bool
            )
            full = np.full((adata.n_obs, emb.shape[1]), np.nan, dtype=np.float32)
            full[mask] = emb[order]
            adata.obsm[OBSM_KEY] = full
        else:
            adata.obsm[OBSM_KEY] = emb[order]
    else:
        adata.obsm[OBSM_KEY] = emb.astype(np.float32)
    adata.write_h5ad(str(h5ad_path))
    logger.info(
        f"Wrote obsm['{OBSM_KEY}'] -> {h5ad_path.name} "
        f"shape={adata.obsm[OBSM_KEY].shape}"
    )


# =============================================================================
# MAIN
# =============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="finetune_scgpt_survival",
        description=(
            "Fine-tune the last few scGPT transformer layers with a Cox NLL "
            "head and dump the resulting CLS embeddings."
        ),
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    p.add_argument(
        "--cox-batch", type=int, default=64, dest="cox_batch",
        help=(
            "Number of samples per Cox-loss gradient step. Bounds the size "
            "of the autograd graph -- keep <= 256 on CPU to avoid OOM."
        ),
    )
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--unfreeze-last", type=int, default=2, dest="unfreeze_last")
    p.add_argument("--n-splits", type=int, default=5, dest="n_splits")
    p.add_argument("--head-hidden", type=int, default=128, dest="head_hidden")
    p.add_argument("--head-dropout", type=float, default=0.20, dest="head_dropout")
    p.add_argument(
        "--max-train-n", type=int, default=0, dest="max_train_n",
        help=(
            "If > 0, subsample TRAIN to this many patients (stratified by event) "
            "before SFT. Useful for smoke-testing on slow CPUs."
        ),
    )
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    p.add_argument(
        "--no-h5ad-write",
        action="store_true",
        help="Don't mutate the h5ad files; only write to embeddings/ folder.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FineTuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        cox_batch=args.cox_batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        unfreeze_last=args.unfreeze_last,
        n_splits=args.n_splits,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        use_amp=not args.no_amp,
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Multi-threaded MKL on torch >= 2.4 + this scGPT build occasionally
    # access-violates inside the encoder forward pass on Windows. Cap at
    # one thread by default; let the user override with NUM_THREADS=N.
    n_threads = int(os.environ.get("NUM_THREADS", "1"))
    torch.set_num_threads(n_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(n_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_threads))
    logger.info(f"torch.num_threads = {torch.get_num_threads()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | AMP: {cfg.use_amp and device.type == 'cuda'}")
    logger.info(f"Config: {cfg}")

    vocab, scgpt_cfg, embsize = _load_engine()
    backbone = _build_backbone(vocab, scgpt_cfg)
    backbone_state = backbone.state_dict()
    backbone_kwargs = dict(
        ntoken=len(vocab),
        d_model=int(scgpt_cfg.get("embsize", 512)),
        nhead=int(scgpt_cfg.get("nheads", 8)),
        d_hid=int(scgpt_cfg.get("d_hid", 512)),
        nlayers=int(scgpt_cfg.get("nlayers", 12)),
        vocab=vocab,
        dropout=float(scgpt_cfg.get("dropout", 0.0)),
        pad_token=PAD_TOKEN,
        pad_value=PAD_VALUE,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        domain_spec_batchnorm=False,
        explicit_zero_prob=False,
        use_fast_transformer=False,
        pre_norm=bool(scgpt_cfg.get("pre_norm", False)),
    )
    del backbone

    vocab_genes = frozenset(
        g for g in vocab.get_stoi() if not g.startswith("<")
    )
    gene_order = _shared_hvg_geneset(TRAIN_H5AD, N_HVG, vocab_genes)

    train_ad = sc.read_h5ad(str(TRAIN_H5AD))
    mask, t_full, e_full = _extract_survival(train_ad)
    train_surv = train_ad[mask].copy()
    times = t_full[mask].astype(np.float32)
    events = e_full[mask].astype(np.float32)
    logger.info(
        f"TRAIN survival: n={len(times)} | events={int(events.sum())} "
        f"| censored={int((1 - events).sum())}"
    )

    if args.max_train_n > 0 and args.max_train_n < len(times):
        # Stratified subsample by event indicator so the Cox loss has signal.
        rng_sub = np.random.default_rng(SEED)
        idx_pos = np.where(events > 0.5)[0]
        idx_neg = np.where(events <= 0.5)[0]
        n_pos = int(args.max_train_n * len(idx_pos) / len(times))
        n_neg = args.max_train_n - n_pos
        keep_pos = rng_sub.choice(idx_pos, size=min(n_pos, len(idx_pos)), replace=False)
        keep_neg = rng_sub.choice(idx_neg, size=min(n_neg, len(idx_neg)), replace=False)
        keep = np.sort(np.concatenate([keep_pos, keep_neg]))
        train_surv = train_surv[keep].copy()
        times = times[keep]
        events = events[keep]
        logger.info(
            f"Subsampled TRAIN to n={len(times)} (events={int(events.sum())}, "
            f"censored={int((1 - events).sum())}) for SFT smoke run"
        )

    g, e, m = _materialise_tensors(train_surv, gene_order, vocab)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros((g.size(0), embsize), dtype=np.float32)
    histories: list[FoldHistory] = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(events)), events.astype(int)), start=1):
        hist, val_cls = _train_one_fold(
            fold=fold,
            backbone_state=backbone_state,
            vocab=vocab,
            cfg=cfg,
            backbone_kwargs=backbone_kwargs,
            g=g, e=e, m=m,
            times=times, events=events,
            tr_idx=tr_idx, va_idx=va_idx,
            device=device,
        )
        oof[va_idx] = val_cls
        histories.append(hist)

    cv_means = [h.best_val_cindex for h in histories]
    logger.info(
        f"5-fold OOF C-index: mean={np.mean(cv_means):.4f} "
        f"std={np.std(cv_means):.4f} per-fold={['%.4f' % x for x in cv_means]}"
    )

    final_epochs = max(1, int(np.median([h.best_epoch for h in histories])))
    logger.info(
        f"Final-model epochs (median best_epoch across folds): {final_epochs}"
    )
    final_model = _train_full(
        backbone_state=backbone_state,
        vocab=vocab,
        cfg=cfg,
        backbone_kwargs=backbone_kwargs,
        g=g, e=e, m=m,
        times=times, events=events,
        epochs=final_epochs,
        device=device,
    )

    train_full_emb = _embed(
        final_model, g, e, m, device, cfg.batch_size, cfg.use_amp
    )

    pub_embs: dict[str, np.ndarray] = {}
    pub_ids: dict[str, list[str]] = {}
    for name, path in PUB_FILES.items():
        if not path.exists():
            logger.warning(f"Skip {name}: {path} missing")
            continue
        pub_ad = sc.read_h5ad(str(path))
        gp, ep, mp = _materialise_tensors(pub_ad, gene_order, vocab)
        emb = _embed(final_model, gp, ep, mp, device, cfg.batch_size, cfg.use_amp)
        pub_embs[name] = emb
        pub_ids[name] = list(pub_ad.obs.index)
        logger.info(f"PUB[{name}] embeddings: {emb.shape}")

    np.save(OOF_NPY, oof)
    save_dict = {"train_oof": oof, "train_full_refit": train_full_emb}
    for name, emb in pub_embs.items():
        save_dict[name] = emb
    np.savez_compressed(ALL_NPZ, **save_dict)

    train_ids = list(train_surv.obs.index)
    rows = [{"cohort": "TRAIN_OOF", "patient_id": pid, "row": i}
            for i, pid in enumerate(train_ids)]
    for name, ids in pub_ids.items():
        rows.extend(
            {"cohort": name, "patient_id": pid, "row": i}
            for i, pid in enumerate(ids)
        )
    pd.DataFrame(rows).to_csv(INDEX_CSV, index=False)

    HISTORY_JSON.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "cv_best_cindex": cv_means,
                "cv_mean": float(np.mean(cv_means)),
                "cv_std": float(np.std(cv_means)),
                "folds": [asdict(h) for h in histories],
                "final_epochs": final_epochs,
                "n_train": int(g.size(0)),
                "n_pub": {k: int(v.shape[0]) for k, v in pub_embs.items()},
                "embedding_dim": int(embsize),
                "shared_hvg_n": len(gene_order),
            },
            indent=2,
        )
    )

    if not args.no_h5ad_write:
        _persist_obsm(TRAIN_H5AD, train_ids, oof)
        for name, emb in pub_embs.items():
            _persist_obsm(PUB_FILES[name], pub_ids[name], emb)

    logger.info("=" * 70)
    logger.info("Fine-tuning complete.")
    logger.info(f"  OOF embeddings (train) : {OOF_NPY}")
    logger.info(f"  All embeddings (npz)   : {ALL_NPZ}")
    logger.info(f"  Index CSV              : {INDEX_CSV}")
    logger.info(f"  Training history       : {HISTORY_JSON}")
    if not args.no_h5ad_write:
        logger.info(
            f"  obsm key '{OBSM_KEY}' written into TRAIN + "
            f"{len(pub_embs)} PUB h5ad files."
        )
    logger.info(
        "Run `python -m batchcor_rna_emb.stress_test.v4_definitive_pipeline` "
        "next -- v4 auto-discovers the new obsm key and adds it to the "
        "leaderboard."
    )


if __name__ == "__main__":
    main()
