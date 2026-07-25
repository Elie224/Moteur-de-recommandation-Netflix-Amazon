"""Tests for Module 4 - PyTorch neural CF (TwoTower, NeuralCF)."""
import numpy as np
import pandas as pd
import pytest

from src.models.neural_cf import TorchCF, TwoTower, NeuralCF


@pytest.fixture
def small_ratings() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id":  [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5],
        "movie_id": [1, 2, 3, 1, 3, 1, 2, 2, 4, 3, 4, 5],
        "rating":   [5, 4, 5, 4, 5, 2, 3, 3, 4, 4, 5, 3],
        "timestamp": list(range(1, 13)),
    })


def test_twotower_shapes():
    m = TwoTower(n_users=5, n_items=4, n_factors=8)
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([0, 1, 2])
    out = m(u, i)
    assert out.shape == (3,)


def test_neuralcf_shapes():
    m = NeuralCF(n_users=5, n_items=4, n_factors=4, hidden=(8,))
    u = torch.tensor([0, 1, 2])
    i = torch.tensor([0, 1, 2])
    out = m(u, i)
    assert out.shape == (3,)


def test_torchcf_fit_predict(small_ratings):
    m = TorchCF(arch="twotower", n_factors=4, n_epochs=2, batch_size=4,
                lr=0.01, reg=0.0).fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 3]})
    preds = m.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0


def test_neuralcf_fit_predict(small_ratings):
    m = TorchCF(arch="neuralcf", n_factors=4, hidden=(8,), n_epochs=3,
                batch_size=4, lr=0.01, reg=0.0).fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 3]})
    preds = m.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0


def test_torchcf_recommend(small_ratings):
    m = TorchCF(arch="twotower", n_factors=4, n_epochs=3, batch_size=4,
                lr=0.01, reg=0.0).fit(small_ratings)
    recs = m.recommend_for_users([1], top_k=3)
    assert 1 in recs
    assert len(recs[1]) >= 1


def test_torchcf_cold_user(small_ratings):
    m = TorchCF(arch="twotower", n_factors=4, n_epochs=2, batch_size=4,
                lr=0.01, reg=0.0).fit(small_ratings)
    recs = m.recommend_for_users([999], top_k=3)
    assert recs.get(999, []) == []


def test_torchcf_loss_decreases(small_ratings):
    m = TorchCF(arch="twotower", n_factors=4, n_epochs=10, batch_size=4,
                lr=0.02, reg=0.0).fit(small_ratings)
    # final should be lower than initial (or at least equal)
    assert m.train_loss_[-1] < m.train_loss_[0] + 1e-3


import torch  # noqa: E402
