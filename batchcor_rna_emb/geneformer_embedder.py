"""Извлечение cell-level embeddings из Geneformer foundation model.

Модуль обеспечивает полный пайплайн: AnnData → h5ad → tokenize → Geneformer EmbExtractor
→ cell-level mean-pool embeddings → PCA-128D → UMAP-3D → сохранение в .obsm/.uns.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Union

import anndata as ad
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.decomposition import PCA
from umap import UMAP

from batchcor_rna_emb.config import (
    BATCH_COL,
    BATCH_SIZE,
    DIAGNOSIS_COL,
    EMB_LABEL_COLS,
    EMBS_LAYER,
    GENEFORMER_EMBEDDING_KEY,
    GENEFORMER_METADATA_KEY,
    GENEFORMER_MODEL_NAME,
    GENEFORMER_N_PCA,
    GENEFORMER_N_UMAP,
    GENEFORMER_PCA_KEY,
    GENEFORMER_UMAP_KEY,
    MAX_CELLS,
    NPROC,
    SEED,
    TOKENIZER_ATTR_DICT,
)


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def tokenize_adata(
    adata: ad.AnnData,
    output_dir: str | Path,
    attr_dict: dict[str, str] = TOKENIZER_ATTR_DICT,
    nproc: int = NPROC,
) -> Path:
    """Токенизирует AnnData и сохраняет HuggingFace dataset на диск.

    Использует ``TranscriptomeTokenizer`` из пакета ``geneformer``.
    AnnData временно сохраняется в h5ad, затем токенизируется.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с raw count или TPM expression в ``.X``
        и ``ensembl_id`` в ``.var_names``.
    output_dir : str or Path
        Директория для сохранения токенизированного датасета.
    attr_dict : dict[str, str]
        Маппинг obs-колонок → имена атрибутов в токенизаторе.
    nproc : int
        Число параллельных воркеров.

    Returns
    -------
    Path
        Путь к папке с HuggingFace dataset (содержит dataset_info.json).

    Raises
    ------
    RuntimeError
        If no dataset folder is found after tokenization.
    """
    from geneformer import TranscriptomeTokenizer

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем AnnData во временный h5ad
    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / "cohort.h5ad"
        adata.write_h5ad(h5ad_path)
        logger.info("Saved temporary h5ad: {}", h5ad_path)

        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=attr_dict,
            nproc=nproc,
        )
        tk.tokenize_data(
            data_directory=str(Path(tmpdir)),
            output_directory=str(output_dir),
            output_prefix="geneformer_tokenized",
            file_format="h5ad",
        )

    dataset_path = _find_dataset_folder(output_dir)
    logger.info("Tokenized dataset saved to: {}", dataset_path)
    return dataset_path


def _find_dataset_folder(token_dir: Path) -> Path:
    """Находит папку HuggingFace dataset внутри директории.

    Parameters
    ----------
    token_dir : Path
        Директория, переданная как output_directory в tokenize_data().

    Returns
    -------
    Path
        Путь к папке с ``dataset_info.json``.

    Raises
    ------
    RuntimeError
        Если папка с датасетом не найдена.
    """
    folders = [
        p for p in token_dir.iterdir()
        if p.is_dir() and (p / "dataset_info.json").exists()
    ]
    if not folders:
        raise RuntimeError(
            f"No HuggingFace dataset folder found in {token_dir}")
    return folders[0]


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def extract_geneformer_embeddings(
    adata: ad.AnnData,
    model_path: Union[str, Path],
    emb_label_cols: list[str] = EMB_LABEL_COLS,
    embs_layer: int = EMBS_LAYER,
    batch_size: int = BATCH_SIZE,
    max_cells: int | None = MAX_CELLS,
    nproc: int = NPROC,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Извлекает per-cell mean-pool embeddings из Geneformer.

    Выполняет токенизацию AnnData во временной директории, затем
    запускает ``EmbExtractor`` для получения cell-level embeddings.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с expression matrix в ``.X``.
    model_path : str or Path
        Путь к директории Geneformer модели
        (например, ``Geneformer-V2-104M_CLcancer``).
    emb_label_cols : list[str]
        Колонки obs (после переименования токенизатором) для metadata.
    embs_layer : int
        Индекс слоя трансформера для извлечения (-1 = последний).
    batch_size : int
        Размер батча для forward pass.
    max_cells : int or None
        Если задан — ограничивает число клеток (для тестов).
    nproc : int
        Число параллельных воркеров для EmbExtractor.

    Returns
    -------
    embeddings : np.ndarray
        Shape ``(n_cells, hidden_dim)`` — float32 embedding matrix.
    metadata : pd.DataFrame
        Shape ``(n_cells, len(emb_label_cols))`` — metadata columns.
    """
    from datasets import load_from_disk
    from geneformer import EmbExtractor

    model_path = Path(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: {}", device)

    with tempfile.TemporaryDirectory() as tmpdir:
        token_dir = Path(tmpdir) / "tokenized"
        output_dir = Path(tmpdir) / "embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Tokenize
        dataset_path = tokenize_adata(adata, token_dir, nproc=nproc)

        # Load and optionally limit
        dataset = load_from_disk(str(dataset_path))
        logger.info(
            "Loaded tokenized dataset: {} cells, columns: {}",
            len(dataset),
            dataset.column_names,
        )

        if max_cells:
            dataset = dataset.select(range(min(max_cells, len(dataset))))
            logger.info("Limited to {} cells for testing", len(dataset))

        # Build EmbExtractor
        emb_extractor = EmbExtractor(
            model_type="CellClassifier",
            num_classes=0,
            emb_mode="cell",
            cell_emb_style="mean_pool",
            filter_data=None,
            max_ncells=max_cells,
            emb_layer=embs_layer,
            emb_label=emb_label_cols,
            labels_to_plot=[],
            forward_batch_size=batch_size,
            nproc=nproc,
        )

        logger.info(
            "Running EmbExtractor on model '{}', layer={}",
            model_path.name,
            embs_layer,
        )
        embs_df = emb_extractor.extract_embs(
            model_directory=str(model_path),
            input_data_file=str(dataset_path),
            output_directory=str(output_dir),
            output_prefix="geneformer_v2",
        )

    logger.info("Raw embs_df shape: {}", embs_df.shape)

    # Разделяем embedding-колонки и metadata
    present_label_cols = [c for c in emb_label_cols if c in embs_df.columns]
    emb_cols = [c for c in embs_df.columns if c not in present_label_cols]

    embeddings = embs_df[emb_cols].values.astype(np.float32)
    metadata = embs_df[present_label_cols]

    logger.info(
        "Geneformer cell-level embeddings: {} samples × {}-dim | metadata columns: {}",
        embeddings.shape[0],
        embeddings.shape[1],
        metadata.columns.tolist(),
    )
    return embeddings, metadata


# ---------------------------------------------------------------------------
# PCA + UMAP reductions  (mirror of compass_embedder.compute_pca_umap_reductions)
# ---------------------------------------------------------------------------

def compute_pca_umap_reductions(
    adata: ad.AnnData,
    embedding_key: str = GENEFORMER_EMBEDDING_KEY,
    n_pca: int = GENEFORMER_N_PCA,
    n_umap: int = GENEFORMER_N_UMAP,
    pca_key: str = GENEFORMER_PCA_KEY,
    umap_key: str = GENEFORMER_UMAP_KEY,
    seed: int = SEED,
) -> None:
    """Вычисляет PCA-128D и UMAP-3D редукции эмбеддингов.

    Результаты сохраняются in-place в ``adata.obsm``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с эмбеддингами в ``.obsm[embedding_key]``.
    embedding_key : str
        Ключ исходных эмбеддингов в ``.obsm``.
    n_pca : int
        Число PCA компонент.
    n_umap : int
        Число UMAP компонент.
    pca_key : str
        Ключ для PCA результата в ``.obsm``.
    umap_key : str
        Ключ для UMAP результата в ``.obsm``.
    seed : int
        Random seed для воспроизводимости.

    Raises
    ------
    KeyError
        If ``embedding_key`` not found in ``.obsm``.
    """
    if embedding_key not in adata.obsm:
        raise KeyError(
            f"Embedding key '{embedding_key}' not found in adata.obsm")

    X = np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    n_samples, n_features = X.shape

    # PCA: ограничиваем n_components до min(n_samples, n_features, n_pca)
    n_pca_actual = min(n_pca, n_samples, n_features)
    logger.info("Computing PCA: {}-dim → {}-dim", n_features, n_pca_actual)

    pca = PCA(n_components=n_pca_actual, random_state=seed)
    pca_embeds = pca.fit_transform(X).astype(np.float32)
    adata.obsm[pca_key] = pca_embeds

    explained_var = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "PCA complete: {}-dim, explained variance ratio: {:.3f}",
        n_pca_actual,
        explained_var,
    )

    # UMAP строим на PCA-reduced данных
    logger.info("Computing UMAP: {}-dim → {}-dim", n_pca_actual, n_umap)
    umap_reducer = UMAP(
        n_components=n_umap,
        random_state=seed,
        n_neighbors=min(15, n_samples - 1),
        min_dist=0.1,
        metric="cosine",
    )
    umap_embeds = umap_reducer.fit_transform(pca_embeds).astype(np.float32)
    adata.obsm[umap_key] = umap_embeds

    logger.info("UMAP complete: {}-dim", n_umap)


# ---------------------------------------------------------------------------
# Metadata storage
# ---------------------------------------------------------------------------

def store_geneformer_metadata(
    adata: ad.AnnData,
    model_path: str,
    embedding_dim: int,
    embs_layer: int = EMBS_LAYER,
    metadata_key: str = GENEFORMER_METADATA_KEY,
) -> None:
    """Сохраняет метаданные Geneformer модели в ``.uns``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData объект (модифицируется in-place).
    model_path : str
        Путь или директория к файлу/папке модели.
    embedding_dim : int
        Размерность итоговых эмбеддингов.
    embs_layer : int
        Индекс слоя, из которого извлечены эмбеддинги.
    metadata_key : str
        Ключ в ``.uns`` для записи метаданных.
    """
    try:
        import geneformer as gf_pkg
        version = gf_pkg.__version__
    except AttributeError:
        version = "unknown"

    metadata = {
        "model_name": "Geneformer",
        "model_version": version,
        "model_type": GENEFORMER_MODEL_NAME,
        "model_source": model_path,
        "embedding_type": "cell_level_mean_pool",
        "embedding_dim": embedding_dim,
        "embs_layer": embs_layer,
        "extraction_method": "EmbExtractor → mean_pool",
        "package": "geneformer",
    }
    adata.uns[metadata_key] = metadata

    logger.info(
        "Stored Geneformer metadata in .uns['{}'], keys: {}",
        metadata_key,
        list(metadata.keys()),
    )


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_geneformer_pipeline(
    adata: ad.AnnData,
    model_path: Union[str, Path],
    emb_label_cols: list[str] = EMB_LABEL_COLS,
    embs_layer: int = EMBS_LAYER,
    batch_size: int = BATCH_SIZE,
    max_cells: int | None = MAX_CELLS,
    nproc: int = NPROC,
    embedding_key: str = GENEFORMER_EMBEDDING_KEY,
    pca_key: str = GENEFORMER_PCA_KEY,
    umap_key: str = GENEFORMER_UMAP_KEY,
    metadata_key: str = GENEFORMER_METADATA_KEY,
    n_pca: int = GENEFORMER_N_PCA,
    n_umap: int = GENEFORMER_N_UMAP,
    seed: int = SEED,
) -> ad.AnnData:
    """Полный пайплайн Geneformer embedding extraction.

    1. Токенизация AnnData → HuggingFace dataset
    2. Извлечение cell-level mean-pool embeddings (EmbExtractor)
    3. PCA-128D редукция
    4. UMAP-3D редукция
    5. Сохранение метаданных в ``.uns``

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с expression matrix в ``.X``.
    model_path : str or Path
        Путь к директории Geneformer модели.
    emb_label_cols : list[str]
        Obs-колонки для переноса в metadata.
    embs_layer : int
        Индекс слоя трансформера (-1 = последний).
    batch_size : int
        Размер батча для EmbExtractor.
    max_cells : int or None
        Ограничение числа клеток (для тестов).
    nproc : int
        Число параллельных воркеров.
    embedding_key : str
        Ключ для raw эмбеддингов в ``.obsm``.
    pca_key : str
        Ключ для PCA-128D в ``.obsm``.
    umap_key : str
        Ключ для UMAP-3D в ``.obsm``.
    metadata_key : str
        Ключ для метаданных в ``.uns``.
    n_pca : int
        Число PCA компонент.
    n_umap : int
        Число UMAP компонент.
    seed : int
        Random seed.

    Returns
    -------
    ad.AnnData
        Модифицированный AnnData с эмбеддингами в ``.obsm``
        и метаданными в ``.uns``.
    """
    model_path = Path(model_path)

    logger.info("=" * 60)
    logger.info("GENEFORMER EMBEDDING PIPELINE START")
    logger.info("Cohort: {} samples × {} genes", adata.n_obs, adata.n_vars)
    logger.info("Model: {}", model_path.name)
    logger.info("=" * 60)

    # 1–2. Токенизация + извлечение эмбеддингов
    embeddings, metadata = extract_geneformer_embeddings(
        adata,
        model_path=model_path,
        emb_label_cols=emb_label_cols,
        embs_layer=embs_layer,
        batch_size=batch_size,
        max_cells=max_cells,
        nproc=nproc,
    )
    adata.obsm[embedding_key] = embeddings

    # Attach metadata columns back to obs (positional alignment preserved by tokenizer)
    for col in metadata.columns:
        adata.obs[f"geneformer_{col}"] = metadata[col].values

    logger.info(
        "Stored embeddings in .obsm['{}']: shape={}",
        embedding_key,
        embeddings.shape,
    )

    # 3–4. PCA + UMAP
    compute_pca_umap_reductions(
        adata,
        embedding_key=embedding_key,
        n_pca=n_pca,
        n_umap=n_umap,
        pca_key=pca_key,
        umap_key=umap_key,
        seed=seed,
    )

    # 5. Метаданные
    store_geneformer_metadata(
        adata,
        model_path=str(model_path),
        embedding_dim=embeddings.shape[1],
        embs_layer=embs_layer,
        metadata_key=metadata_key,
    )

    logger.info("=" * 60)
    logger.info("GENEFORMER EMBEDDING PIPELINE COMPLETE")
    logger.info(
        "Keys added: .obsm[{}], .obsm[{}], .obsm[{}], .uns[{}]",
        embedding_key,
        pca_key,
        umap_key,
        metadata_key,
    )
    logger.info("=" * 60)

    return adata
