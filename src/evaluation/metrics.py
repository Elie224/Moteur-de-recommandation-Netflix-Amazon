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

Design choices (documented)
---------------------------
- **Relevance threshold**: by default ratings >= 4 are considered *relevant*
  (MovieLens convention, used in the literature).
- **Users with no relevant test item are excluded** from ranking averages.
  The metric is undefined (always zero) for such users, including them would
  dilute the signal. Use ``include_empty_users=True`` to override.
- **Binary vs graded relevance for NDCG** is controlled by ``graded``:
    - ``graded=False`` (default): binary relevance (1 if rating >= threshold).
    - ``graded=True``         : graded, the actual rating is used.
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
    """Convert an explicit rating into a binary relevance label."""
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
    include_empty_users: bool = False,
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
        Ratings >= threshold are considered relevant (used for binary NDCG
        and for the relevance sets of P/R/HR/MAP).
    graded : bool
        If True, NDCG uses the actual rating as graded relevance instead of
        a binary 1/0 label.
    include_empty_users : bool
        If True, average over all users in ``recommendations`` (users with no
        relevant test item contribute 0 to all metrics). Default False (only
        users with at least one relevant item are averaged - cleaner signal).

    Notes
    -----
    The recommendation list MUST NOT contain items the user already saw in
    train. Callers are responsible for excluding them (the model wrappers do
    this via ``exclude_seen=True``).
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
    uids = recommendations.keys() if include_empty_users else [
        uid for uid in recommendations if user_rel.get(uid)
    ]
    for uid in uids:
        recs = recommendations[uid]
        rel = user_rel.get(uid, set())
        user_ratings = user_graded.get(uid, {})

        p_list.append(precision_at_k(recs, rel, k))
        r_list.append(recall_at_k(recs, rel, k))
        hr_list.append(hit_rate_at_k(recs, rel, k))
        ap_list.append(average_precision_at_k(recs, rel, k))

        if graded:
            ndcg_relevance = {iid: r for iid, r in user_ratings.items()
                               if r >= relevance_threshold}
        else:
            ndcg_relevance = {iid: 1.0 for iid in rel}
        ndcg_list.append(ndcg_at_k(recs, ndcg_relevance, k))

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
    surfaced: set[int] = set()
    for recs in recommendations.values():
        surfaced.update(recs[:k])
    return len(surfaced) / max(len(catalog), 1)


def novelty(
    recommendations: dict[int, list[int]],
    item_popularity: dict[int, int],
    k: int = 10,
) -> float:
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


def evaluate_topk(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    n_users: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """End-to-end top-K evaluation with strict candidate policy.

    For each test user we ask the model for ``top_k=k`` recommendations
    on the **entire catalog** and verify the model excludes train items.
    The same user population is used for every model so results are
    directly comparable.

    Parameters
    ----------
    model : object
        Any model with ``recommend_for_users(user_ids, top_k)``.
    train, test : DataFrame
        Long-format ratings with columns ``user_id``, ``movie_id``, ``rating``.
    k : int
        Cut-off.
    relevance_threshold : float
        Rating threshold for relevance.
    n_users : int | None
        Subsample of test users (random with ``seed``). None = all.
    """
    train_users = set(train["user_id"].unique())
    test_eval = test[test["user_id"].isin(train_users)].copy()
    rng = np.random.default_rng(seed)
    uids = test_eval["user_id"].unique()
    if n_users is not None and n_users < len(uids):
        uids = rng.choice(uids, size=n_users, replace=False)
    uids = list(uids)
    test_subset = test_eval[test_eval["user_id"].isin(uids)]
    recs = model.recommend_for_users(uids, top_k=k)
    metrics = evaluate_ranking(
        recs, test_subset, k=k, relevance_threshold=relevance_threshold,
    )
    return metrics
