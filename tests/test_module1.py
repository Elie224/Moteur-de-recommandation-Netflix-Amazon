"""End-to-end smoke test for Module 1.

Uses a synthetic mini-dataset so the test runs fast and offline.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    catalog_coverage,
    evaluate_ranking,
    evaluate_rating_prediction,
    novelty,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)
from src.models import ItemItemCosine, PopularityBaseline


@pytest.fixture
def small_ratings() -> pd.DataFrame:
    """Tiny dense-ish ratings: 5 users x 6 movies.

    User 1 rates {1,2,3,6}, so has 2 unseen items (4,5).
    User 5 rates {2,4,6}, so has 3 unseen items.
    """
    data = {
        "user_id":  [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5],
        "movie_id": [1, 2, 3, 6, 1, 3, 5, 1, 4, 2, 5, 2, 4, 6],
        "rating":   [5, 4, 5, 3, 4, 5, 4, 2, 5, 3, 4, 5, 4, 5],
        "timestamp": list(range(1, 15)),
    }
    return pd.DataFrame(data)


def test_popularity_baseline(small_ratings):
    pop = PopularityBaseline(min_ratings=1, score="count").fit(small_ratings)
    recs = pop.recommend_for_users([5], top_k=3)[5]
    # User 5 has rated {2, 4, 6} -> top-3 popularity among unseen
    assert len(recs) == 3
    # None of user 5's already-rated items should appear
    assert set(recs).isdisjoint({2, 4, 6})


def test_cosine_fit_recommend(small_ratings):
    model = ItemItemCosine().fit(small_ratings)
    # Use user 5 which has 3 unseen items out of 6
    recs = model.recommend_for_users([5], top_k=3)
    assert 5 in recs
    assert len(recs[5]) == 3
    # Should not recommend items user 5 already rated
    assert set(recs[5]).isdisjoint({2, 4, 6})


def test_cosine_cold_user(small_ratings):
    model = ItemItemCosine().fit(small_ratings)
    # User 999 doesn't exist in train -> empty recommendation
    recs = model.recommend_for_users([999], top_k=5)
    assert recs == {} or recs.get(999, []) == []


def test_cosine_predict_clipped(small_ratings):
    model = ItemItemCosine().fit(small_ratings)
    pairs = pd.DataFrame({"user_id": [1, 2], "movie_id": [1, 2]})
    pairs["rating"] = [5.0, 4.0]
    preds = model.predict_for_pairs(pairs)
    for v in preds.values():
        assert 1.0 <= v <= 5.0


def test_metrics_basic():
    recs = [1, 2, 3, 4, 5]
    rel = {2, 4}
    assert precision_at_k(recs, rel, k=5) == pytest.approx(0.4)
    assert recall_at_k(recs, rel, k=5) == pytest.approx(1.0)
    n = ndcg_at_k(recs, {2: 5.0, 4: 3.0}, k=5)
    assert 0.0 < n <= 1.0


def test_evaluate_ranking_shapes(small_ratings):
    train = small_ratings.iloc[:10]
    test = small_ratings.iloc[10:]
    model = ItemItemCosine().fit(train)
    recs = model.recommend_for_users(test["user_id"].unique().tolist(), top_k=5)
    metrics = evaluate_ranking(recs, test, k=5)
    for k in ["precision@5", "recall@5", "hit_rate@5", "ndcg@5", "map@5"]:
        assert k in metrics


def test_catalog_coverage_and_novelty():
    recs = {1: [1, 2, 3], 2: [4, 5, 6]}
    cat = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    cov = catalog_coverage(recs, cat, k=3)
    assert 0.0 <= cov <= 1.0
    nov = novelty(recs, {1: 100, 2: 10, 3: 1, 4: 50, 5: 5, 6: 1}, k=3)
    assert nov >= 0.0
