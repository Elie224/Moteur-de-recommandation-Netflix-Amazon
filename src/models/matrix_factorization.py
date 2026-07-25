"""Matrix Factorization implementations from scratch.

Reference: Yehuda Koren, Robert Bell, Chris Volinsky,
"Matrix Factorization Techniques for Recommender Systems", IEEE Computer 2009.

Two implementations are provided:

1. FunkSVD (a.k.a. SVD with biases, SGD-based)
   - User factors P (n_users x k)
   - Item factors Q (n_items x k)
   - User biases bu (n_users)
   - Item biases bi (n_items)
   - Global mean mu
   - Prediction: r_hat = mu + bu[u] + bi[i] + P[u] @ Q[i]
   - Loss: MSE + L2 reg on P, Q, bu, bi

2. ALS-WR (Alternating Least Squares with Weighted Regularization)
   - Closed-form updates for biases and factors, alternating.
   - Faster convergence on large data, no learning rate tuning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# FunkSVD
# ---------------------------------------------------------------------------


@dataclass
class FunkSVD:
    """Matrix factorization with biases, trained by mini-batch SGD."""

    n_factors: int = 50
    n_epochs: int = 20
    lr: float = 0.005
    reg: float = 0.02
    reg_bias: float = 0.02
    batch_size: int = 8192
    random_state: int = 0
    verbose: bool = False

    # filled by fit()
    user_index_: dict[int, int] = field(default_factory=dict)
    item_index_: dict[int, int] = field(default_factory=dict)
    P_: np.ndarray | None = None
    Q_: np.ndarray | None = None
    bu_: np.ndarray | None = None
    bi_: np.ndarray | None = None
    mu_: float = 0.0
    train_loss_: list[float] = field(default_factory=list)
    val_rmse_: list[float] = field(default_factory=list)

    def fit(
        self,
        ratings: pd.DataFrame,
        val: pd.DataFrame | None = None,
    ) -> "FunkSVD":
        rng = np.random.default_rng(self.random_state)
        user_ids = np.sort(ratings["user_id"].unique())
        item_ids = np.sort(ratings["movie_id"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(user_ids)}
        self.item_index_ = {int(m): i for i, m in enumerate(item_ids)}
        n_u, n_i = len(user_ids), len(item_ids)

        u_idx = ratings["user_id"].map(self.user_index_).to_numpy()
        i_idx = ratings["movie_id"].map(self.item_index_).to_numpy()
        r = ratings["rating"].to_numpy(np.float64)

        # Init parameters
        self.P_ = rng.normal(0, 0.01, size=(n_u, self.n_factors))
        self.Q_ = rng.normal(0, 0.01, size=(n_i, self.n_factors))
        self.bu_ = np.zeros(n_u, dtype=np.float64)
        self.bi_ = np.zeros(n_i, dtype=np.float64)
        self.mu_ = float(r.mean())

        if val is not None:
            val_u = val["user_id"].map(self.user_index_).to_numpy()
            val_i = val["movie_id"].map(self.item_index_).to_numpy()
            val_r = val["rating"].to_numpy(np.float64)
            # Filter rows with unknown user/item (cold rows)
            mask = (
                np.isin(val["user_id"].to_numpy(), list(self.user_index_.keys()))
                & np.isin(val["movie_id"].to_numpy(), list(self.item_index_.keys()))
            )
            val_u = val_u[mask]; val_i = val_i[mask]; val_r = val_r[mask]

        n = len(r)
        idx_all = np.arange(n)

        for epoch in range(self.n_epochs):
            rng.shuffle(idx_all)
            epoch_loss = 0.0
            for start in range(0, n, self.batch_size):
                batch = idx_all[start:start + self.batch_size]
                ub = u_idx[batch]
                ib = i_idx[batch]
                rb = r[batch]

                # Predictions
                pred = self.mu_ + self.bu_[ub] + self.bi_[ib] + np.einsum(
                    "ij,ij->i", self.P_[ub], self.Q_[ib]
                )
                err = rb - pred

                # Update biases
                self.bu_[ub] += self.lr * (err - self.reg_bias * self.bu_[ub])
                self.bi_[ib] += self.lr * (err - self.reg_bias * self.bi_[ib])

                # Update factors
                pu = self.P_[ub]
                qi = self.Q_[ib]
                self.P_[ub] += self.lr * (err[:, None] * qi - self.reg * pu)
                self.Q_[ib] += self.lr * (err[:, None] * pu - self.reg * qi)

                epoch_loss += float((err ** 2).sum())

            rmse_train = float(np.sqrt(epoch_loss / n))
            self.train_loss_.append(rmse_train)
            if val is not None:
                val_pred = (
                    self.mu_
                    + self.bu_[val_u]
                    + self.bi_[val_i]
                    + np.einsum("ij,ij->i", self.P_[val_u], self.Q_[val_i])
                )
                val_rmse = float(np.sqrt(np.mean((val_pred - val_r) ** 2)))
                self.val_rmse_.append(val_rmse)
            if self.verbose:
                msg = f"  epoch {epoch+1:3d}/{self.n_epochs}  train RMSE {rmse_train:.4f}"
                if val is not None:
                    msg += f"  val RMSE {val_rmse:.4f}"
                print(msg)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _predict(self, uid: int, iid: int) -> float:
        if uid not in self.user_index_ or iid not in self.item_index_:
            return self.mu_
        u = self.user_index_[uid]
        i = self.item_index_[iid]
        pred = self.mu_ + self.bu_[u] + self.bi_[i] + self.P_[u] @ self.Q_[i]
        return float(np.clip(pred, 1.0, 5.0))

    def predict_for_pairs(self, pairs: pd.DataFrame) -> dict[tuple[int, int], float]:
        return {(int(r.user_id), int(r.movie_id)): self._predict(int(r.user_id), int(r.movie_id))
                for r in pairs.itertuples()}

    def all_user_predictions(self, user_id: int) -> np.ndarray:
        if user_id not in self.user_index_:
            return np.array([])
        u = self.user_index_[user_id]
        scores = self.mu_ + self.bu_[u] + self.bi_ + self.P_[u] @ self.Q_.T
        return np.clip(scores, 1.0, 5.0)

    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_seen: bool = True,
        seen_items: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if user_id not in self.user_index_:
            return []
        scores = self.all_user_predictions(user_id)
        if exclude_seen:
            seen = seen_items or set()
            for iid in seen:
                if iid in self.item_index_:
                    scores[self.item_index_[iid]] = -np.inf
        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = np.array(sorted(self.item_index_.keys()))
        return [(int(item_ids[i]), float(scores[i])) for i in top_idx if scores[i] != -np.inf]

    def recommend_for_users(self, user_ids, top_k: int = 10) -> dict[int, list[int]]:
        return {int(uid): [iid for iid, _ in self.recommend(int(uid), top_k=top_k)]
                for uid in user_ids}


# ---------------------------------------------------------------------------
# ALS-WR (Alternating Least Squares, weighted regularization)
# ---------------------------------------------------------------------------


@dataclass
class ALSMF:
    """ALS with biases + factors. Faster convergence than SGD on dense data."""

    n_factors: int = 50
    n_epochs: int = 15
    reg: float = 0.05
    reg_bias: float = 0.02
    verbose: bool = False

    user_index_: dict[int, int] = field(default_factory=dict)
    item_index_: dict[int, int] = field(default_factory=dict)
    n_users_: int = 0
    n_items_: int = 0
    R_: np.ndarray | None = None  # sparse? we keep dense for ALS
    P_: np.ndarray | None = None
    Q_: np.ndarray | None = None
    bu_: np.ndarray | None = None
    bi_: np.ndarray | None = None
    mu_: float = 0.0
    train_rmse_: list[float] = field(default_factory=list)

    def _dense_matrix(self, u_idx, i_idx, r, n_u, n_i):
        R = np.full((n_u, n_i), np.nan, dtype=np.float64)
        R[u_idx, i_idx] = r
        return R

    def fit(self, ratings: pd.DataFrame, val: pd.DataFrame | None = None) -> "ALSMF":
        user_ids = np.sort(ratings["user_id"].unique())
        item_ids = np.sort(ratings["movie_id"].unique())
        self.user_index_ = {int(u): i for i, u in enumerate(user_ids)}
        self.item_index_ = {int(m): i for i, m in enumerate(item_ids)}
        self.n_users_ = len(user_ids)
        self.n_items_ = len(item_ids)

        u_idx = ratings["user_id"].map(self.user_index_).to_numpy()
        i_idx = ratings["movie_id"].map(self.item_index_).to_numpy()
        r = ratings["rating"].to_numpy(np.float64)
        self.mu_ = float(r.mean())

        R = self._dense_matrix(u_idx, i_idx, r, self.n_users_, self.n_items_)

        # Biases closed form (Tikhonov)
        # bu_u = sum_i (r_ui - mu) / (lambda + |I_u|)
        # bi_i = sum_u (r_ui - mu - bu_u) / (lambda + |U_i|)
        self.bu_ = np.zeros(self.n_users_)
        self.bi_ = np.zeros(self.n_items_)
        counts_u = (~np.isnan(R)).sum(axis=1)
        counts_i = (~np.isnan(R)).sum(axis=0)

        for u in range(self.n_users_):
            mask = ~np.isnan(R[u])
            if mask.any():
                self.bu_[u] = np.nansum(R[u] - self.mu_) / (self.reg_bias + counts_u[u])
        for i in range(self.n_items_):
            mask = ~np.isnan(R[:, i])
            if mask.any():
                residuals = R[:, i] - self.mu_ - self.bu_
                self.bi_[i] = np.nansum(residuals) / (self.reg_bias + counts_i[i])

        # Build residual matrix
        residuals = R - (self.mu_ + self.bu_[:, None] + self.bi_[None, :])
        residuals[np.isnan(residuals)] = 0.0

        # Init factors
        rng = np.random.default_rng(0)
        self.P_ = rng.normal(0, 0.01, (self.n_users_, self.n_factors))
        self.Q_ = rng.normal(0, 0.01, (self.n_items_, self.n_factors))

        # ALS: alternate P and Q updates
        I_u = (counts_u > 0).astype(np.float64)
        I_i = (counts_i > 0).astype(np.float64)
        eye_k = self.reg * np.eye(self.n_factors)

        for epoch in range(self.n_epochs):
            # Update P: P = (Q^T Q + reg I)^-1 Q^T R_resid
            # Per-user, weighted by I_u (we only consider items user rated)
            QtQ = self.Q_.T @ self.Q_ + eye_k
            for u in range(self.n_users_):
                if counts_u[u] == 0:
                    continue
                # Items user u rated
                rated_items = np.where(~np.isnan(R[u]))[0]
                Qi = self.Q_[rated_items]
                ri = residuals[u, rated_items]
                A = Qi.T @ Qi + self.reg * np.eye(self.n_factors)
                b = Qi.T @ ri
                self.P_[u] = np.linalg.solve(A, b)

            # Update Q similarly
            for i in range(self.n_items_):
                if counts_i[i] == 0:
                    continue
                rated_users = np.where(~np.isnan(R[:, i]))[0]
                Pu = self.P_[rated_users]
                ri = residuals[rated_users, i]
                A = Pu.T @ Pu + self.reg * np.eye(self.n_factors)
                b = Pu.T @ ri
                self.Q_[i] = np.linalg.solve(A, b)

            # RMSE on observed ratings
            pred = self.mu_ + self.bu_[:, None] + self.bi_[None, :] + self.P_ @ self.Q_.T
            obs = ~np.isnan(R)
            rmse = float(np.sqrt(np.nanmean((R[obs] - pred[obs]) ** 2)))
            self.train_rmse_.append(rmse)
            if self.verbose:
                print(f"  ALS epoch {epoch+1}/{self.n_epochs}  RMSE {rmse:.4f}")
        return self

    def _predict(self, uid, iid):
        if uid not in self.user_index_ or iid not in self.item_index_:
            return self.mu_
        u = self.user_index_[uid]; i = self.item_index_[iid]
        return float(np.clip(self.mu_ + self.bu_[u] + self.bi_[i] + self.P_[u] @ self.Q_[i], 1, 5))

    def predict_for_pairs(self, pairs):
        return {(int(r.user_id), int(r.movie_id)): self._predict(int(r.user_id), int(r.movie_id))
                for r in pairs.itertuples()}

    def all_user_predictions(self, user_id):
        if user_id not in self.user_index_:
            return np.array([])
        u = self.user_index_[user_id]
        scores = self.mu_ + self.bu_[u] + self.bi_ + self.P_[u] @ self.Q_.T
        return np.clip(scores, 1.0, 5.0)

    def recommend(self, user_id, top_k=10, exclude_seen=True, seen_items=None):
        if user_id not in self.user_index_:
            return []
        scores = self.all_user_predictions(user_id)
        if exclude_seen and seen_items:
            for iid in seen_items:
                if iid in self.item_index_:
                    scores[self.item_index_[iid]] = -np.inf
        k = min(top_k, scores.size)
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        item_ids = np.array(sorted(self.item_index_.keys()))
        return [(int(item_ids[i]), float(scores[i])) for i in top_idx if scores[i] != -np.inf]

    def recommend_for_users(self, user_ids, top_k=10):
        return {int(uid): [iid for iid, _ in self.recommend(int(uid), top_k=top_k)]
                for uid in user_ids}
