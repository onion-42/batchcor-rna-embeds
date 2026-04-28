"""Извлечение geneset-level bottleneck эмбеддингов из COMPASS foundation model.

Модуль обеспечивает полный пайплайн: AnnData → TPM DataFrame → COMPASS model.project()
→ flattened geneset-level embeddings → PCA-128D → UMAP-3D → сохранение в .obsm/.uns.
"""
from __future__ import annotations

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
    COMPASS_EMBEDDING_KEY,
    COMPASS_METADATA_KEY,
    COMPASS_MODEL_URL,
    COMPASS_N_PCA,
    COMPASS_N_UMAP,
    COMPASS_PCA_KEY,
    COMPASS_UMAP_KEY,
    SEED,
)


def build_tpm_for_compass(
    adata: ad.AnnData,
    cancer_code: str,
) -> pd.DataFrame:
    """Конвертирует AnnData expression matrix в формат COMPASS.

    COMPASS ожидает DataFrame, где первый столбец — cancer type code (TCGA),
    а остальные столбцы — гены (TPM-нормализованная экспрессия).

    Parameters
    ----------
    adata : ad.AnnData
        AnnData объект с TPM экспрессией в ``.X``.
    cancer_code : str
        TCGA cancer type abbreviation (e.g. ``"SKCM"``, ``"KIRC"``).

    Returns
    -------
    pd.DataFrame
        DataFrame shape ``(n_obs, 1 + n_vars)`` с cancer code в первом столбце.

    Raises
    ------
    ValueError
        If ``adata.X`` is None.
    """
    if adata.X is None:
        raise ValueError("adata.X is None — cannot extract TPM expression")

    from scipy import sparse as sp

    x_dense: np.ndarray
    if sp.issparse(adata.X):
        x_dense = np.asarray(adata.X.toarray(), dtype=np.float32)
    else:
        x_dense = np.asarray(adata.X, dtype=np.float32)

    df_tpm = pd.DataFrame(
        data=x_dense,
        index=adata.obs_names,
        columns=adata.var_names,
        dtype=np.float32,
    )

    # COMPASS требует cancer code в первом столбце
    df_tpm.insert(0, "cancer_type", cancer_code)

    logger.info(
        "Built COMPASS TPM input: {} samples × {} genes, cancer_code='{}'",
        df_tpm.shape[0],
        df_tpm.shape[1] - 1,
        cancer_code,
    )
    return df_tpm


def _flatten_geneset_embeddings(
    dfgs: pd.DataFrame,
    patient_names: pd.Index,
    n_genesets: int,
) -> np.ndarray:
    """Flatten per-patient geneset vector embeddings в единый вектор.

    ``model.project()`` возвращает DataFrame с multi-index вида
    ``"patient_id$$geneset_name"`` и shape ``(n_patients * n_genesets, n_channels)``.
    Здесь мы reshape в ``(n_patients, n_genesets * n_channels)``.

    Parameters
    ----------
    dfgs : pd.DataFrame
        Geneset-level vector features from ``model.project()``.
        Shape ``(n_patients * n_genesets, n_channels)``.
    patient_names : pd.Index
        Original patient/sample names.
    n_genesets : int
        Number of genesets in COMPASS (typically 133).

    Returns
    -------
    np.ndarray
        Flattened embeddings, shape ``(n_patients, n_genesets * n_channels)``,
        dtype ``float32``.
    """
    n_patients = len(patient_names)
    n_channels = dfgs.shape[1]
    expected_rows = n_patients * n_genesets

    if dfgs.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in dfgs "
            f"({n_patients} patients × {n_genesets} genesets), "
            f"got {dfgs.shape[0]}"
        )

    # Reshape: (n_patients * n_genesets, n_channels) → (n_patients, n_genesets * n_channels)
    flat = dfgs.values.reshape(n_patients, n_genesets * n_channels)

    logger.debug(
        "Flattened geneset embeddings: {} × {} → {} × {}",
        n_patients,
        f"{n_genesets}×{n_channels}",
        flat.shape[0],
        flat.shape[1],
    )
    return flat.astype(np.float32)


def extract_compass_embeddings(
    adata: ad.AnnData,
    model: object,
    cancer_code: str,
    batch_size: int = 128,
) -> np.ndarray:
    """Извлекает geneset-level bottleneck embeddings из COMPASS.

    Использует ``model.project()`` для получения geneset-level vector features
    (до concept bottleneck), затем flatten в единый вектор.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с TPM в ``.X``.
    model : object
        Загруженная COMPASS модель (результат ``loadcompass()``).
    cancer_code : str
        TCGA cancer type code.
    batch_size : int
        Batch size для inference.

    Returns
    -------
    np.ndarray
        Flattened geneset embeddings, shape ``(n_obs, n_genesets * n_channels)``,
        dtype ``float32``.
    """
    df_tpm = build_tpm_for_compass(adata, cancer_code)

    logger.info("Running COMPASS model.project() with batch_size={}", batch_size)
    dfgs, _dfct = model.project(df_tpm, batch_size=batch_size)

    # Определяем количество genesets из модели
    n_genesets = len(model.model.geneset_feature_name)
    logger.info(
        "COMPASS project output: dfgs shape={}, n_genesets={}, n_channels={}",
        dfgs.shape,
        n_genesets,
        dfgs.shape[1],
    )

    embeddings = _flatten_geneset_embeddings(dfgs, adata.obs_names, n_genesets)

    logger.info(
        "COMPASS geneset-level embeddings: {} samples × {}-dim",
        embeddings.shape[0],
        embeddings.shape[1],
    )
    return embeddings


def compute_pca_umap_reductions(
    adata: ad.AnnData,
    embedding_key: str = COMPASS_EMBEDDING_KEY,
    n_pca: int = COMPASS_N_PCA,
    n_umap: int = COMPASS_N_UMAP,
    pca_key: str = COMPASS_PCA_KEY,
    umap_key: str = COMPASS_UMAP_KEY,
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
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.obsm")

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

    # UMAP: строим на PCA-reduced данных
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


def store_compass_metadata(
    adata: ad.AnnData,
    model_version: str,
    model_path: str,
    cancer_code: str,
    embedding_dim: int,
    metadata_key: str = COMPASS_METADATA_KEY,
) -> None:
    """Сохраняет метаданные COMPASS модели в ``.uns``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData объект (модифицируется in-place).
    model_version : str
        Версия COMPASS модели (e.g. ``"2.3"``).
    model_path : str
        Путь или URL к файлу модели.
    cancer_code : str
        Использованный TCGA cancer code.
    embedding_dim : int
        Размерность итоговых эмбеддингов.
    metadata_key : str
        Ключ в ``.uns`` для записи метаданных.
    """
    metadata = {
        "model_name": "COMPASS",
        "model_version": model_version,
        "model_type": "finetuner_pft_all",
        "model_source": model_path,
        "cancer_code_used": cancer_code,
        "embedding_type": "geneset_level_flattened",
        "embedding_dim": embedding_dim,
        "extraction_method": "model.project() → flatten geneset-level",
        "package": "immuno-compass",
    }
    adata.uns[metadata_key] = metadata

    logger.info(
        "Stored COMPASS metadata in .uns['{}'], keys: {}",
        metadata_key,
        list(metadata.keys()),
    )


def load_compass_model(
    model_path: Union[str, Path],
    device: str = "cpu",
) -> object:
    """Загружает COMPASS модель из файла.

    Parameters
    ----------
    model_path : str or Path
        Путь к ``.pt`` файлу модели.
    device : str
        Устройство: ``"cpu"`` или ``"cuda"``.

    Returns
    -------
    object
        Загруженная COMPASS модель (FineTuner / PreTrainer).

    Raises
    ------
    FileNotFoundError
        If model file does not exist.
    """
    from compass import loadcompass

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"COMPASS model not found: {model_path}")

    logger.info("Loading COMPASS model from '{}' to device='{}'", model_path, device)
    model = loadcompass(str(model_path), map_location=device)

    # Получаем версию пакета
    try:
        import compass as compass_pkg
        version = compass_pkg.__version__
    except AttributeError:
        version = "unknown"

    logger.info("COMPASS model loaded, package version={}", version)
    return model


def run_compass_pipeline(
    adata: ad.AnnData,
    model_path: Union[str, Path],
    cancer_code: str,
    device: str = "cpu",
    batch_size: int = 128,
    embedding_key: str = COMPASS_EMBEDDING_KEY,
    pca_key: str = COMPASS_PCA_KEY,
    umap_key: str = COMPASS_UMAP_KEY,
    metadata_key: str = COMPASS_METADATA_KEY,
    n_pca: int = COMPASS_N_PCA,
    n_umap: int = COMPASS_N_UMAP,
    seed: int = SEED,
) -> ad.AnnData:
    """Полный пайплайн COMPASS embedding extraction.

    1. Загрузка модели
    2. Извлечение geneset-level bottleneck embeddings
    3. PCA-128D редукция
    4. UMAP-3D редукция
    5. Сохранение метаданных в ``.uns``

    Parameters
    ----------
    adata : ad.AnnData
        AnnData с TPM expression в ``.X``.
    model_path : str or Path
        Путь к COMPASS ``.pt`` модели.
    cancer_code : str
        TCGA cancer type code.
    device : str
        Устройство для inference (``"cpu"`` / ``"cuda"``).
    batch_size : int
        Batch size для model.project().
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
        Модифицированный AnnData с эмбеддингами в ``.obsm`` и метаданными в ``.uns``.
    """
    logger.info("=" * 60)
    logger.info("COMPASS EMBEDDING PIPELINE START")
    logger.info("Cohort: {} samples × {} genes", adata.n_obs, adata.n_vars)
    logger.info("Cancer code: {}", cancer_code)
    logger.info("=" * 60)

    # 1. Загрузка модели
    model = load_compass_model(model_path, device=device)

    # 2. Извлечение эмбеддингов
    embeddings = extract_compass_embeddings(
        adata, model, cancer_code, batch_size=batch_size,
    )
    adata.obsm[embedding_key] = embeddings

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
    try:
        import compass as compass_pkg
        version = compass_pkg.__version__
    except AttributeError:
        version = "unknown"

    store_compass_metadata(
        adata,
        model_version=version,
        model_path=str(model_path),
        cancer_code=cancer_code,
        embedding_dim=embeddings.shape[1],
        metadata_key=metadata_key,
    )

    logger.info("=" * 60)
    logger.info("COMPASS EMBEDDING PIPELINE COMPLETE")
    logger.info(
        "Keys added: .obsm[{}], .obsm[{}], .obsm[{}], .uns[{}]",
        embedding_key,
        pca_key,
        umap_key,
        metadata_key,
    )
    logger.info("=" * 60)

    return adata
