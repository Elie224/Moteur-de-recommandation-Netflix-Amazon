"""Tests for Module 2 - Surprise wrappers."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Surprise wants ~/.surprise_data which we may not have access to on Windows
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(Path.cwd() / ".surprise_data"))

from src.models.surprise_models import make_surprise_model


@pytest.fixture
def small_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
        "movie_id": [1, 2, 3, 1, 3, 1, 2, 4, 2, 4, 1, 4],
        "rating":   [5, 4, 5, 4, 5, 2, 3, 5, 3, 4, 1, 5],
        "timestamp": list(range(1, 13)),
    })


@pytest.mark.parametrize("algo,kwargs", [
    ("BaselineOnly", {}),
    ("SVD", {"n_factors": 5, "n_epochs": 5, "random_state": 0}),
    ("NMF", {"n_factors": 5, "n_epochs": 5, "random_state": 0}),
    ("CoClustering", {"n_epochs": 5, "random_state": 0}),
    ("KNNWithMeans", {"k": 2, "sim_options": {"name": "pearson_baseline", "user_based": True, "min_support": 1}}),
])
def test_surprise_models_fit_predict(small_ratings, algo, kwargs):
    m = make_surprise_model(algo, **kwargs).fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 2]})
    preds = m.predict_for_pairs(pairs)
    assert len(preds) == 2
    for v in preds.values():
        assert 1.0 <= v <= 5.0


def test_surprise_recommend(small_ratings):
    m = make_surprise_model("SVD", n_factors=5, n_epochs=5, random_state=0).fit(small_ratings)
    recs = m.recommend_for_users([1], top_k=3)
    assert 1 in recs
    assert len(recs[1]) == 3


def test_surprise_cold_user(small_ratings):
    m = make_surprise_model("SVD", n_factors=5, n_epochs=5, random_state=0).fit(small_ratings)
    recs = m.recommend_for_users([999], top_k=3)
    assert recs.get(999, []) == [] or 999 not in recs
