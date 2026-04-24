"""Model training wrappers: TabPFN and LightAutoML."""
from __future__ import annotations

import numpy as np
from loguru import logger
from sklearn.base import BaseEstimator


def train_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
) -> BaseEstimator:
    """
    Train a TabPFN classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    seed : int
        Random seed.

    Returns
    -------
    BaseEstimator
        Fitted TabPFN classifier.
    """
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier(random_state=seed)
    model.fit(X_train, y_train)
    logger.info("TabPFN fitted: {} samples x {} features", X_train.shape[0], X_train.shape[1])
    return model


def train_lama(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
    timeout: int = 300,
    task_name: str = "binary",
) -> BaseEstimator:
    """
    Train a LightAutoML model.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    seed : int
        Random seed.
    timeout : int
        Training timeout in seconds.
    task_name : str
        Task type: ``'binary'`` or ``'multiclass'``.

    Returns
    -------
    BaseEstimator
        Fitted LightAutoML model with ``predict_proba`` method.
    """
    import pandas as pd
    from lightautoml.automl.presets.tabular_presets import TabularAutoML
    from lightautoml.tasks import Task

    task = Task(task_name)
    automl = TabularAutoML(
        task=task,
        timeout=timeout,
        random_state=seed,
    )

    df_train = pd.DataFrame(X_train, columns=[f"f_{i}" for i in range(X_train.shape[1])])
    df_train["target"] = y_train

    automl.fit_predict(df_train, roles={"target": "target"})
    logger.info(
        "LightAutoML fitted: {} samples x {} features, timeout={}s",
        X_train.shape[0], X_train.shape[1], timeout,
    )
    return automl


def predict_proba(
    model: BaseEstimator,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Get positive-class probabilities from a fitted model.

    Handles both sklearn-style models and LightAutoML.

    Parameters
    ----------
    model : BaseEstimator
        Fitted classifier.
    X_test : np.ndarray
        Test feature matrix.

    Returns
    -------
    np.ndarray
        Predicted probabilities for positive class, shape ``(n_samples,)``.
    """
    # LightAutoML uses predict() which returns DataFrame
    if hasattr(model, "predict") and not hasattr(model, "predict_proba"):
        import pandas as pd
        df_test = pd.DataFrame(X_test, columns=[f"f_{i}" for i in range(X_test.shape[1])])
        preds = model.predict(df_test)
        return np.asarray(preds).ravel()

    # sklearn-style: predict_proba returns (n_samples, n_classes)
    proba = model.predict_proba(X_test)
    if proba.ndim == 2:
        return proba[:, 1]
    return proba.ravel()
