"""Evaluation metrics for recommender systems.

We support two families:

1. Rating prediction:
    - RMSE (root mean squared error)
    - MAE  (mean absolute error)

2. Ranking / top-K recommendation:
    - Precision@K
    - Recall@K
    - Hit Rate@K
    - MAP@K (mean average precision)
    - NDCG@K (normalized discounted cumulative gain)

All metrics operate on DataFrames in the long format:
    user_id, item_id, rating, [timestamp], [prediction]
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rating prediction metrics
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _relevance_threshold(rating: float, threshold: float) -> int:
    """Convert an explicit rating into a binary relevance label.

    Following common practice (and the MovieLens literature), a rating is
    considered *relevant* if it is >= ``threshold`` (default 4 in MovieLens).
    """
    return 1 if rating >= threshold else 0


def evaluate_rating_prediction(
    test: pd.DataFrame, predictions: dict[tuple[int, int], float]
) -> dict[str, float]:
    """Compute RMSE / MAE between ``predictions`` and ``test`` ratings."""
    y_true = []
    y_pred = []
    for _, row in test.iterrows():
        key = (int(row["user_id"]), int(row["movie_id"]))
        if key in predictions:
            y_true.append(row["rating"])
            y_pred.append(predictions[key])
    if not y_true:
        return {"rmse": float("nan"), "mae": float("nan"), "coverage": 0.0}
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    coverage = len(y_true) / max(len(test), 1)
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "coverage": float(coverage),
    }


# ---------------------------------------------------------------------------
# Top-K ranking metrics
# ---------------------------------------------------------------------------


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return sum(1 for r in rec_k if r in relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    return sum(1 for r in rec_k if r in relevant) / len(relevant)


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    rec_k = recommended[:k]
    return float(any(r in relevant for r in rec_k))


def average_precision_at_k(
    recommended: list[int], relevant: set[int], k: int
) -> float:
    if not relevant:
        return 0.0
    rec_k = recommended[:k]
    hits = 0
    score = 0.0
    for i, item in enumerate(rec_k, start=1):
        if item in relevant:
            hits += 1
            score += hits / i
    return score / min(len(relevant), k)


def ndcg_at_k(recommended: list[int], relevant: dict[int, float], k: int) -> float:
    """``relevant`` maps item_id -> graded relevance (e.g. rating)."""
    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k, start=1):
        rel = relevant.get(item, 0.0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(i + 1)

    # ideal DCG: sort true relevances descending
    ideal_rels = sorted(relevant.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_rels, start=1):
        idcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregator over many users
# ---------------------------------------------------------------------------


def evaluate_ranking(
    recommendations: dict[int, list[int]],
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    graded: bool = False,
) -> dict[str, float]:
    """Aggregate ranking metrics over all users with at least one test item.

    Parameters
    ----------
    recommendations : dict
        Mapping ``user_id -> ranked list of recommended item_ids`` (already
        sorted best-first).
    test : DataFrame
        Test ratings with columns ``user_id``, ``movie_id``, ``rating``.
    k : int
        Cut-off for top-K metrics.
    relevance_threshold : float
        Ratings >= threshold are considered relevant (only for ungraded eval).
    graded : bool
        If True, use NDCG with the actual rating as graded relevance.
    """
    user_rel: dict[int, set[int]] = defaultdict(set)
    user_graded: dict[int, dict[int, float]] = defaultdict(dict)
    for _, row in test.iterrows():
        uid = int(row["user_id"])
        iid = int(row["movie_id"])
        rating = float(row["rating"])
        if rating >= relevance_threshold:
            user_rel[uid].add(iid)
        user_graded[uid][iid] = rating

    p_list, r_list, hr_list, ap_list, ndcg_list = [], [], [], [], []
    for uid, recs in recommendations.items():
        rel = user_rel.get(uid, set())
        graded = user_graded.get(uid, {})
        # only score users that have at least one relevant test item (otherwise
        # the metric is ill-defined / always zero).
        if graded:
            p_list.append(precision_at_k(recs, rel, k))
            r_list.append(recall_at_k(recs, rel, k))
            hr_list.append(hit_rate_at_k(recs, rel, k))
            ap_list.append(average_precision_at_k(recs, rel, k))
            ndcg_list.append(
                ndcg_at_k(recs, {iid: r for iid, r in graded.items() if r >= relevance_threshold}, k)
            )

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    return {
        f"precision@{k}": _mean(p_list),
        f"recall@{k}": _mean(r_list),
        f"hit_rate@{k}": _mean(hr_list),
        f"map@{k}": _mean(ap_list),
        f"ndcg@{k}": _mean(ndcg_list),
        "n_users_evaluated": len(p_list),
    }


def catalog_coverage(
    recommendations: dict[int, list[int]],
    catalog: set[int],
    k: int = 10,
) -> float:
    """Fraction of the catalog that appears in the top-K across all users."""
    surfaced: set[int] = set()
    for recs in recommendations.values():
        surfaced.update(recs[:k])
    return len(surfaced) / max(len(catalog), 1)


def novelty(
    recommendations: dict[int, list[int]],
    item_popularity: dict[int, int],
    k: int = 10,
) -> float:
    """Mean -log2(pop / total_interactions) over recommended items.

    Higher novelty = recommending more long-tail / less popular items.
    """
    total = sum(item_popularity.values())
    if total == 0:
        return 0.0
    scores = []
    for recs in recommendations.values():
        for item in recs[:k]:
            p = item_popularity.get(item, 0) / total
            if p > 0:
                scores.append(-math.log2(p))
    return float(np.mean(scores)) if scores else 0.0
