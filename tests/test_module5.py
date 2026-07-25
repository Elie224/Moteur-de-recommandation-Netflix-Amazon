"""Tests for Module 5 - LightFM-style hybrid model from scratch."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.models.hybrid import HybridCF, LightFMModel


@pytest.fixture
def small_users():
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "gender": ["M", "F", "M", "F"],
        "age": [25, 35, 18, 45],
        "occupation": [1, 5, 7, 2],
        "zip_code": ["0", "0", "0", "0"],
    })


@pytest.fixture
def small_movies():
    return pd.DataFrame({
        "movie_id": [1, 2, 3, 4, 5],
        "title": ["a", "b", "c", "d", "e"],
        "genres": ["Action|Drama", "Comedy", "Drama", "Action|Thriller", "Comedy|Drama"],
        "year": [1990, 2000, 1985, 2010, 1995],
        "genres_list": [["Action", "Drama"], ["Comedy"], ["Drama"], ["Action", "Thriller"], ["Comedy", "Drama"]],
    })


@pytest.fixture
def small_ratings():
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2, 3, 3, 4, 4, 4],
        "movie_id": [1, 2, 3, 1, 3, 1, 2, 3, 4, 5],
        "rating": [5, 4, 5, 4, 5, 2, 3, 4, 5, 4],
        "timestamp": list(range(1, 11)),
    })


def test_hybrid_model_shapes():
    # 0 reserved for padding; real features start at 1
    m = HybridCF(n_users=5, n_items=4, n_user_features=10, n_item_features=8, n_factors=4)
    u = torch.tensor([0, 1])
    i = torch.tensor([0, 1])
    uf = torch.tensor([[0, 1, 0, 0], [2, 3, 0, 0]])
    if_ = torch.tensor([[0, 0, 0], [1, 0, 0]])
    out = m(u, i, uf, if_)
    assert out.shape == (2,)


def test_lightfm_build_features(small_users, small_movies):
    m = LightFMModel(n_factors=4)
    m.build_user_features(small_users)
    m.build_item_features(small_movies)
    assert m.user_features_mat_.shape[0] == 4
    assert m.item_features_mat_.shape[0] == 5
    assert len(m.user_feat_index_) > 0
    assert len(m.item_feat_index_) > 0


def test_lightfm_fit_mse(small_users, small_movies, small_ratings):
    m = LightFMModel(n_factors=4, n_epochs=3, batch_size=4, lr=0.01, reg=0.0)
    m.build_user_features(small_users)
    m.build_item_features(small_movies)
    m.fit(small_ratings)
    assert m.model is not None
    assert len(m.train_loss_) == 3


def test_lightfm_recommend(small_users, small_movies, small_ratings):
    m = LightFMModel(n_factors=4, n_epochs=2, batch_size=4, lr=0.01, reg=0.0)
    m.build_user_features(small_users)
    m.build_item_features(small_movies)
    m.fit(small_ratings)
    recs = m.recommend_for_users([1], top_k=2)
    assert 1 in recs
    assert len(recs[1]) >= 1


def test_lightfm_cold_user(small_users, small_movies, small_ratings):
    """A brand-new user (no history) should still get recommendations via features."""
    m = LightFMModel(n_factors=4, n_epochs=2, batch_size=4, lr=0.01, reg=0.0)
    m.build_user_features(small_users)
    m.build_item_features(small_movies)
    m.fit(small_ratings)
    recs = m.recommend_for_new_user(
        user_features=[("gender", "F"), ("age", "25"), ("occupation", "1")],
        top_k=3,
    )
    assert len(recs) >= 1


def test_lightfm_predict_clipped(small_users, small_movies, small_ratings):
    m = LightFMModel(n_factors=4, n_epochs=2, batch_size=4, lr=0.01, reg=0.0)
    m.build_user_features(small_users)
    m.build_item_features(small_movies)
    m.fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 3]})
    preds = m.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0
