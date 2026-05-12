"""
Export each h5ad in ``data/processed/`` as a slim, share-ready AnnData
containing ONLY:

* ``obs.index``           — patient_id (preserved verbatim from the source)
* a small whitelist of ``obs`` columns (Cohort, Diagnosis, Therapy_group,
  PFS / PFS_FLAG / OS / OS_FLAG / Response, etc. — drops free-text fields
  and Kassandra / MFP scores so the file stays small and tidy)
* every embedding stored in ``obsm`` whose KEY MATCHES the whitelist
  below: ``scGPT_embedding``, ``cAE_embedding``, ``cAE_embedding_OOD``
  (PCA128d_* and UMAP3d_* are skipped — easy for others to recompute and
  cuts file size roughly in half).

The gene-expression matrix ``X``, ``var``, ``layers``, and ``uns`` are
ALL dropped — they're identical for everyone in the consortium and add
hundreds of MB for no reason at merge time.

Output layout::

    data/exports/embeddings_to_merge/
        TRAIN_Combined_cAE_Corrected.h5ad   ← smallest possible TRAIN
        KIRC_Tissue_ICI_Pred.h5ad
        ...
        MANIFEST.txt                        ← shapes + SHA-256 per file
        SHARE.md                            ← merge instructions for others
        embeddings_to_merge.zip             ← single archive (Telegram-ready)

Run::

    python scripts/export_embeddings_for_merge.py
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    level="INFO",
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO_ROOT      : Path = Path(__file__).resolve().parents[1]
SRC_DIR        : Path = REPO_ROOT / "data" / "processed"
OUT_DIR        : Path = REPO_ROOT / "data" / "exports" / "embeddings_to_merge"
ZIP_PATH       : Path = OUT_DIR.with_suffix(".zip")
MANIFEST_PATH  : Path = OUT_DIR / "MANIFEST.txt"
SHARE_MD_PATH  : Path = OUT_DIR / "SHARE.md"

OBSM_KEEP: tuple[str, ...] = (
    "scGPT_embedding",          # raw 512-D from frozen scGPT
    "cAE_embedding",             # cAE-corrected 512-D (group 7's contribution)
    "cAE_embedding_OOD",         # cAE on cohorts unseen during training
    "scGPT_finetuned_embedding", # only present if you ran finetune_scgpt_survival
)

# Compact, broadly-useful obs columns. Anything missing is silently skipped.
OBS_KEEP: tuple[str, ...] = (
    "Cohort", "Diagnosis", "Therapy_group", "Pat_Condition_MSKCC",
    "Stage", "Gender", "Age", "TMB", "PDL1_TC_IHC_num",
    "PFS", "PFS_FLAG", "OS", "OS_FLAG",
    "Response", "Responder", "RECIST",   # one of these is usually present
    "BOR", "Benefit",                     # alternate response labels
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _human_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _slim(adata: sc.AnnData, src: Path) -> sc.AnnData:
    """Return a copy with only patient_id, whitelisted obs, and embeddings."""
    keep_obs = [c for c in OBS_KEEP if c in adata.obs.columns]
    keep_obsm = [k for k in OBSM_KEEP if k in adata.obsm]
    if not keep_obsm:
        raise RuntimeError(
            f"{src.name}: no whitelisted embedding keys in obsm "
            f"(have: {list(adata.obsm.keys())})"
        )

    obs_df = adata.obs[keep_obs].copy() if keep_obs else pd.DataFrame(index=adata.obs.index)

    # 0-column var keeps AnnData happy without storing any gene data
    var_df = pd.DataFrame(index=pd.Index([], name="gene"))
    X      = np.zeros((adata.n_obs, 0), dtype=np.float32)

    slim = ad.AnnData(X=X, obs=obs_df, var=var_df)
    for k in keep_obsm:
        slim.obsm[k] = np.asarray(adata.obsm[k], dtype=np.float32)

    slim.uns["bg7_source_file"]   = src.name
    slim.uns["bg7_source_n_obs"]  = int(adata.n_obs)
    slim.uns["bg7_export_date"]   = datetime.now().isoformat(timespec="seconds")
    slim.uns["bg7_embedding_keys"] = list(keep_obsm)
    return slim


def _write_share_md() -> None:
    """Drop a SHARE.md alongside the slim h5ads explaining how to merge."""
    rows = []
    for h5 in sorted(OUT_DIR.glob("*.h5ad")):
        a = sc.read_h5ad(str(h5))
        emb_keys = sorted(a.obsm.keys())
        shapes = {k: tuple(a.obsm[k].shape) for k in emb_keys}
        rows.append(
            f"- **{h5.name}** — n={a.n_obs}, obs.cols={list(a.obs.columns)}, "
            f"obsm shapes={shapes}"
        )
    listing = "\n".join(rows) if rows else "(no files exported yet)"

    SHARE_MD_PATH.write_text(
        f"""# Group 7 — embeddings drop ({datetime.now():%Y-%m-%d})

This folder ships slimmed `.h5ad` files containing **only patient IDs +
clinical labels + scGPT / cAE embeddings**, ready to merge with the other
groups' exports.

The gene-expression matrix `X`, `var`, `layers` and `uns` from the source
files were intentionally dropped — they are identical for everyone in the
consortium and add hundreds of MB at zero merge value.

## Files

{listing}

## How to merge with another group's drop

```python
import scanpy as sc
import anndata as ad

# 1. Load every contributor's slim h5ad
parts = [
    sc.read_h5ad("group7/TRAIN_Combined_cAE_Corrected.h5ad"),
    sc.read_h5ad("group3/TRAIN_with_their_embeddings.h5ad"),
    # ... etc
]

# 2. Inner-join on patient_id and keep all embedding obsm keys
merged = ad.concat(
    parts,
    axis=0,                  # stack patients (same patients across groups -> use axis=1)
    join="outer",
    label="contributor",     # adds an obs col flagging where each row came from
    merge="first",           # keep first-seen value for shared obs cols
    uns_merge="first",
)

# 3. Sanity-check the embeddings are aligned
for key in merged.obsm:
    print(key, merged.obsm[key].shape)
```

If every group's drop covers the **same patients** but adds **different
embedding keys** (e.g. group 3 ships `scGPT_finetuned_embedding`, group 7
ships `cAE_embedding`), use `axis=1` style merging on `obs.index` instead:

```python
import functools, anndata as ad

def join_on_patient(left: ad.AnnData, right: ad.AnnData) -> ad.AnnData:
    shared = left.obs.index.intersection(right.obs.index)
    left, right = left[shared].copy(), right[shared].copy()
    for k, v in right.obsm.items():
        if k not in left.obsm:
            left.obsm[k] = v
    return left

merged = functools.reduce(join_on_patient, parts)
```

## Provenance

Generated by `scripts/export_embeddings_for_merge.py`. See `MANIFEST.txt`
in this folder for per-file `n_obs`, embedding shapes and SHA-256
checksums you can paste into your message to the receiving team.
""",
        encoding="utf-8",
    )


def _write_manifest() -> None:
    lines: list[str] = []
    lines.append(f"BG-Internship Group 7 — embeddings drop")
    lines.append(f"Generated:    {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Source dir:   {SRC_DIR}")
    lines.append(f"Output dir:   {OUT_DIR}")
    lines.append("")
    lines.append(f"{'file':<46} {'size_MB':>8}  {'n_obs':>6}  {'embeddings':<40} sha256")
    lines.append("-" * 130)
    for h5 in sorted(OUT_DIR.glob("*.h5ad")):
        a = sc.read_h5ad(str(h5))
        emb_keys = sorted(a.obsm.keys())
        shapes = ",".join(f"{k}{tuple(a.obsm[k].shape)}" for k in emb_keys)
        lines.append(
            f"{h5.name:<46} {_human_mb(h5):>8.1f}  {a.n_obs:>6}  {shapes:<40} {_sha256(h5)}"
        )
    MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"Source dir not found: {SRC_DIR}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_files = sorted(SRC_DIR.glob("*.h5ad"))
    if not src_files:
        raise SystemExit(f"No .h5ad files under {SRC_DIR}")

    logger.info(f"Slimming {len(src_files)} files from {SRC_DIR}")
    total_in_mb  = 0.0
    total_out_mb = 0.0
    for src in src_files:
        size_in = _human_mb(src)
        total_in_mb += size_in
        logger.info(f"  reading {src.name} ({size_in:.1f} MB) ...")
        try:
            adata = sc.read_h5ad(str(src))
            slim  = _slim(adata, src)
        except Exception as exc:
            logger.warning(f"    skipping {src.name}: {exc}")
            continue
        dst = OUT_DIR / src.name
        slim.write_h5ad(str(dst), compression="gzip")
        size_out = _human_mb(dst)
        total_out_mb += size_out
        logger.info(
            f"    -> {dst.name}  n={slim.n_obs}  "
            f"obsm={sorted(slim.obsm.keys())}  ({size_out:.1f} MB, "
            f"{(1 - size_out / max(size_in, 1e-9)) * 100:.0f}% smaller)"
        )

    _write_share_md()
    _write_manifest()

    logger.success(
        f"Wrote {len(list(OUT_DIR.glob('*.h5ad')))} h5ad + MANIFEST.txt + SHARE.md "
        f"to {OUT_DIR} (total {total_out_mb:.1f} MB from {total_in_mb:.1f} MB source)"
    )

    # Single archive that's easy to upload anywhere
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in sorted(OUT_DIR.iterdir()):
            if f.is_file():
                zf.write(f, arcname=f.name)
    logger.success(f"Bundled archive: {ZIP_PATH} ({_human_mb(ZIP_PATH):.1f} MB)")
    logger.info("Next steps:")
    logger.info(f"  1. Upload `{ZIP_PATH.name}` to Drive / Slack / Telegram")
    logger.info(f"  2. Paste the SHA-256 lines from MANIFEST.txt in the message")
    logger.info(f"  3. Receiving team: `unzip embeddings_to_merge.zip` + see SHARE.md")


if __name__ == "__main__":
    main()
