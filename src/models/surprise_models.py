"""Wrappers around scikit-surprise algorithms with a unified API.

All wrappers expose:
    .fit(ratings_df)            -> self
    .predict_for_pairs(pairs)   -> dict[(uid, iid) -> predicted_rating]
    .recommend(uid, top_k)      -> list[(iid, score)]
    .recommend_for_users(...)   -> dict[uid -> list[iid]]

Supported algorithms:
    - KNNWithMeans   (user-user or item-item cosine / pearson / msd)
    - SVD            (classic Funk SVD with biases)
    - NMF            (non-negative matrix factorization)
    - BaselineOnly   (ALS bias model)
    - CoClustering   (co-clustering)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from surprise import (
    AlgoBase,
    BaselineOnly,
    CoClustering,
    Dataset,
    KNNWithMeans,
    NMF,
    Reader,
    SVD,
    accuracy,
)
from surprise.model_selection import train_test_split as surprise_train_test_split


class SurpriseWrapper:
    """Unified wrapper around a surprise algorithm."""

    def __init__(self, algo: AlgoBase, name: Optional[str] = None) -> None:
        self.algo = algo
        self.name = name or algo.__class__.__name__

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, ratings: pd.DataFrame) -> "SurpriseWrapper":
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            ratings[["user_id", "movie_id", "rating"]], reader
        )
        self.trainset_ = data.build_full_trainset()
        self.algo.fit(self.trainset_)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_for_pairs(self, pairs: pd.DataFrame) -> dict[tuple[int, int], float]:
        out: dict[tuple[int, int], float] = {}
        for _, row in pairs.iterrows():
            uid = int(row["user_id"])
            iid = int(row["movie_id"])
            pred = self.algo.predict(uid, iid, verbose=False)
            out[(uid, iid)] = float(pred.est)
        return out

    def predict_for_user_item(self, user_id: int, item_id: int) -> float:
        return float(self.algo.predict(int(user_id), int(item_id), verbose=False).est)

    # ------------------------------------------------------------------
    # Recommend
    # ------------------------------------------------------------------
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        seen_items: Optional[set[int]] = None,
    ) -> list[tuple[int, float]]:
        try:
            inner_uid = self.trainset_.to_inner_uid(int(user_id))
            user_seen_inner = self.trainset_.ur[inner_uid]
        except ValueError:
            return []  # cold user

        all_iids = list(self.trainset_.all_items())
        scored = []
        for inner_iid in all_iids:
            raw_iid = self.trainset_.to_raw_iid(inner_iid)
            if exclude_seen and seen_items is not None and int(raw_iid) in seen_items:
                continue
            est = self.algo.predict(
                int(user_id), int(raw_iid), verbose=False
            ).est
            scored.append((int(raw_iid), float(est)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def recommend_for_users(
        self, user_ids: list[int], top_k: int = 10
    ) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for uid in user_ids:
            recs = self.recommend(uid, top_k=top_k)
            out[int(uid)] = [iid for iid, _ in recs]
        return out

    # ------------------------------------------------------------------
    # All-ratings utilities
    # ------------------------------------------------------------------
    def all_user_predictions(self, user_id: int) -> np.ndarray:
        """Return a vector of predicted ratings for every item in trainset."""
        try:
            self.trainset_.to_inner_uid(int(user_id))
        except ValueError:
            return np.array([])
        all_iids = [self.trainset_.to_raw_iid(i) for i in self.trainset_.all_items()]
        preds = [
            self.algo.predict(int(user_id), int(iid), verbose=False).est
            for iid in all_iids
        ]
        return np.asarray(preds)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_surprise_model(
    algo: str = "SVD",
    **kwargs,
) -> SurpriseWrapper:
    """Build a SurpriseWrapper for the requested algorithm.

    Supported ``algo`` values:
        - "SVD"
        - "NMF"
        - "BaselineOnly"
        - "CoClustering"
        - "KNNWithMeans"
    """
    algo = algo.lower()
    if algo == "svd":
        model = SVD(**kwargs)
    elif algo == "nmf":
        model = NMF(**kwargs)
    elif algo == "baselineonly":
        model = BaselineOnly(**kwargs)
    elif algo == "coclustering":
        model = CoClustering(**kwargs)
    elif algo == "knnwithmeans":
        # Defaults to user-based cosine (classic CF)
        defaults = {
            "k": 40,
            "min_k": 1,
            "sim_options": {
                "name": "pearson_baseline",
                "user_based": True,
                "min_support": 1,
            },
        }
        defaults.update(kwargs)
        model = KNNWithMeans(**defaults)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    return SurpriseWrapper(model, name=algo)
