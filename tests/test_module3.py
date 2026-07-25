"""Tests for Module 3 - FunkSVD + ALS from scratch."""
import numpy as np
import pandas as pd
import pytest

from src.models.matrix_factorization import FunkSVD, ALSMF


@pytest.fixture
def small_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 5],
        "movie_id": [1, 2, 3, 4, 1, 3, 2, 4, 5, 1, 4, 2, 3, 4, 5],
        "rating":   [5, 4, 5, 3, 4, 5, 3, 4, 5, 2, 4, 5, 4, 5, 3],
        "timestamp": list(range(1, 16)),
    })


def test_funksvd_fit_shapes(small_ratings):
    m = FunkSVD(n_factors=5, n_epochs=2, lr=0.01, random_state=0).fit(small_ratings)
    assert m.P_.shape == (5, 5)
    assert m.Q_.shape == (5, 5)
    assert m.bu_.shape == (5,)
    assert m.bi_.shape == (5,)
    assert m.mu_ > 0


def test_funksvd_predict_clipped(small_ratings):
    m = FunkSVD(n_factors=5, n_epochs=2, lr=0.01, random_state=0).fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [1, 3, 5]})
    preds = m.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0


def test_funksvd_recommend(small_ratings):
    m = FunkSVD(n_factors=5, n_epochs=3, lr=0.01, random_state=0).fit(small_ratings)
    recs = m.recommend_for_users([1], top_k=3)
    assert 1 in recs
    assert len(recs[1]) >= 1  # user 1 has seen many items


def test_funksvd_cold_user(small_ratings):
    m = FunkSVD(n_factors=5, n_epochs=2, lr=0.01, random_state=0).fit(small_ratings)
    recs = m.recommend_for_users([999], top_k=5)
    assert recs.get(999, []) == []


def test_funksvd_loss_decreases(small_ratings):
    m = FunkSVD(n_factors=5, n_epochs=10, lr=0.02, reg=0.01, random_state=0).fit(small_ratings)
    assert m.train_loss_[-1] < m.train_loss_[0]


def test_als_fit_shapes(small_ratings):
    m = ALSMF(n_factors=5, n_epochs=2, reg=0.05).fit(small_ratings)
    assert m.P_.shape == (5, 5)
    assert m.Q_.shape == (5, 5)


def test_als_predict_clipped(small_ratings):
    m = ALSMF(n_factors=5, n_epochs=2, reg=0.05).fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2, 3], "movie_id": [1, 3, 5]})
    preds = m.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0
