"""Pre-train production models and save standardized, self-contained artifacts.

Conventions (P6):
    artifacts/
        item_item_cosine.pkl
        surprise_svd.pkl
        hybrid_lightfm_state.pt
        movies.csv
        metadata.json
        manifest.json
        user_index.json
        item_index.json
        user_features.npy
        item_features.npy

Run once before starting the API / dashboard:
    .venv/Scripts/python scripts/train_production_models.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_movielens_1m
from src.models import ItemItemCosine
from src.models.hybrid import LightFMModel
from src.models.surprise_models import make_surprise_model

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "processed" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / ".surprise_data"))

    print("[1/6] Loading MovieLens 1M...")
    ml = load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
    train = ml.ratings
    print(f"  {len(train):,} ratings, {ml.n_users:,} users, {ml.n_items:,} items")

    print("[2/6] Training ItemItemCosine...")
    cos = ItemItemCosine(top_neighbours=200).fit(train)
    with open(ART / "item_item_cosine.pkl", "wb") as f:
        pickle.dump(cos, f)
    print(f"  saved {ART/'item_item_cosine.pkl'}")

    print("[3/6] Training Surprise SVD (50 factors, 20 epochs)...")
    svd = make_surprise_model("SVD", n_factors=50, n_epochs=20,
                              lr_all=0.005, reg_all=0.02, random_state=0).fit(train)
    with open(ART / "surprise_svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    print(f"  saved {ART/'surprise_svd.pkl'}")

    print("[4/6] Training LightFM hybrid (PyTorch, 32 factors, 8 epochs)...")
    lfm = LightFMModel(n_factors=32, n_epochs=8, batch_size=8192,
                       lr=0.005, reg=1e-5, loss="mse")
    lfm.build_user_features(ml.users)
    lfm.build_item_features(ml.movies)
    lfm.fit(train)

    # Save self-contained cold-start artifacts (P10)
    state = {
        "model_state": lfm.model.state_dict(),
        "n_user_feats": len(lfm.user_feat_index_) + 1,  # +1 for padding
        "n_item_feats": len(lfm.item_feat_index_) + 1,
        "n_factors": lfm.n_factors,
    }
    torch.save(state, ART / "hybrid_lightfm_state.pt")

    # Build the feature matrices in the *rating* order (matches training)
    rating_user_ids = np.sort(train["user_id"].unique())
    rating_item_ids = np.sort(train["movie_id"].unique())
    lfm.user_index_ = {int(u): i for i, u in enumerate(rating_user_ids)}
    lfm.item_index_ = {int(m): i for i, m in enumerate(rating_item_ids)}

    users_df = ml.users.set_index("user_id").loc[rating_user_ids].reset_index()
    movies_df = ml.movies.set_index("movie_id").loc[rating_item_ids].reset_index()

    def _build_user_mat():
        keys_per_row = []
        for _, row in users_df.iterrows():
            keys_per_row.append([
                ("gender", str(row["gender"])),
                ("age", str(row["age"])),
                ("occupation", str(row["occupation"])),
            ])
        max_n = max(len(k) for k in keys_per_row)
        mat = np.zeros((len(keys_per_row), max_n), dtype=np.int64)
        for i, ks in enumerate(keys_per_row):
            for j, k in enumerate(ks):
                if k in lfm.user_feat_index_:
                    mat[i, j] = lfm.user_feat_index_[k]
        return mat

    def _build_item_mat():
        keys_per_row = []
        for _, row in movies_df.iterrows():
            ks = [("genre", g) for g in (row["genres_list"] or [])]
            try:
                year = int(row["year"]) if row["year"] == row["year"] else None
            except (TypeError, ValueError):
                year = None
            if year:
                ks.append(("decade", str((year // 10) * 10)))
            keys_per_row.append(ks)
        max_n = max(len(k) for k in keys_per_row) if keys_per_row else 1
        mat = np.zeros((len(keys_per_row), max_n), dtype=np.int64)
        for i, ks in enumerate(keys_per_row):
            for j, k in enumerate(ks):
                if k in lfm.item_feat_index_:
                    mat[i, j] = lfm.item_feat_index_[k]
        return mat

    np.save(ART / "user_features.npy", _build_user_mat())
    np.save(ART / "item_features.npy", _build_item_mat())
    json.dump(lfm.user_index_, open(ART / "user_index.json", "w"))
    json.dump(lfm.item_index_, open(ART / "item_index.json", "w"))
    json.dump(
        {f"{k[0]}={k[1]}": v for k, v in lfm.user_feat_index_.items()},
        open(ART / "user_feat_vocab.json", "w"),
    )
    json.dump(
        {f"{k[0]}={k[1]}": v for k, v in lfm.item_feat_index_.items()},
        open(ART / "item_feat_vocab.json", "w"),
    )
    print(f"  saved {ART/'hybrid_lightfm_state.pt'} + feature mats + mappings")

    print("[5/6] Saving movies metadata...")
    movies_meta = ml.movies[["movie_id", "title", "year", "genres_list"]].copy()
    movies_meta["genres_str"] = movies_meta["genres_list"].apply(
        lambda g: "|".join(g) if isinstance(g, list) else ""
    )
    movies_meta = movies_meta.drop(columns=["genres_list"])
    movies_meta.to_csv(ART / "movies.csv", index=False)

    print("[6/6] Writing manifest + metadata...")
    metadata = {
        "n_users": int(ml.n_users),
        "n_items": int(ml.n_items),
        "n_ratings": int(ml.n_ratings),
        "n_user_features": len(lfm.user_feat_index_),
        "n_item_features": len(lfm.item_feat_index_),
    }
    json.dump(metadata, open(ART / "metadata.json", "w"), indent=2)

    manifest = {
        "dataset_version": "ml-1m",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "models": {
            "item_item_cosine":  {"path": "item_item_cosine.pkl",       "version": "1.0.0"},
            "surprise_svd":      {"path": "surprise_svd.pkl",           "version": "1.0.0"},
            "hybrid_lightfm":     {"path": "hybrid_lightfm_state.pt",   "version": "1.0.0"},
        },
        "config": {
            "cosine_top_neighbours": 200,
            "svd_n_factors": 50, "svd_n_epochs": 20,
            "lfm_n_factors": 32, "lfm_n_epochs": 8, "lfm_loss": "mse",
        },
    }
    json.dump(manifest, open(ART / "manifest.json", "w"), indent=2)
    print(f"  saved manifest.json")

    # Cleanup legacy names if present
    for legacy in ["cosine.pkl", "svd.pkl", "lightfm.pt", "meta.json"]:
        legacy_path = ART / legacy
        if legacy_path.exists():
            try:
                legacy_path.unlink()
                print(f"  removed legacy {legacy}")
            except OSError:
                pass

    print("\nDone. Artifacts in:", ART)


if __name__ == "__main__":
    main()
