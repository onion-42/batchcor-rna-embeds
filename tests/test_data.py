"""Tests for cohort merge utilities and response label binarisation."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from batchcor_rna_emb.batch_correction.run_cae_correction import concatenate_cohorts
from batchcor_rna_emb.stress_test.v4_definitive_pipeline import (
    binarise_labels,
    detect_response_column,
)


def test_concatenate_cohorts_preserves_obs_and_var():
    X = np.random.RandomState(1).rand(3, 5).astype(np.float32)
    a1 = ad.AnnData(X=X, obs=pd.DataFrame({"cohort": ["A"] * 3}, index=[f"A_{i}" for i in range(3)]))
    a2 = ad.AnnData(
        X=np.random.RandomState(2).rand(2, 5).astype(np.float32),
        obs=pd.DataFrame({"cohort": ["B"] * 2}, index=[f"B_{i}" for i in range(2)]),
    )
    a1.var_names = a2.var_names = [f"g{i}" for i in range(5)]

    joint = concatenate_cohorts([a1, a2])
    assert joint.n_obs == 5
    assert joint.n_vars == 5
    assert list(joint.obs["cohort"].values) == ["A", "A", "A", "B", "B"]


def test_detect_response_column_prefers_named_column():
    # ``'response'`` is a substring of ``response_*`` but not of ``responder``,
    # so priority ranks ``response_flag`` above ``responder`` when both exist.
    obs = pd.DataFrame(
        {
            "foo": [1, 2],
            "responder": ["yes", "no"],
            "response_flag": [0, 1],
        }
    )
    col = detect_response_column(obs)
    assert col == "response_flag"


def test_detect_response_column_single_responder():
    obs = pd.DataFrame({"responder": ["yes", "no"]})
    assert detect_response_column(obs) == "responder"


def test_binarise_labels_positive_negative_lexicon():
    s = pd.Series(["CR", "nr", "PR", "pd"], index=list("abcd"))
    out = binarise_labels(s)
    assert out is not None
    bin_, mapping = out
    assert set(bin_.unique().tolist()) == {0, 1}
    assert mapping["CR"] == 1
    assert mapping["nr"] == 0


def test_binarise_labels_rejects_single_class():
    s = pd.Series(["yes", "yes", "yes"])
    assert binarise_labels(s) is None
