"""Pre-train production models and save artifacts for the API.

Trains three models on MovieLens 1M (full data) and saves them under
data/processed/artifacts/. Run once before starting the API:

    .venv/Scripts/python scripts/train_production_models.py

Artifacts saved:
    - cosine.joblib               : ItemItemCosine (pickled)
    - svd_trained.pkl             : Surprise SVD trained model
    - lightfm_hybrid.pt           : LightFM (hybrid) PyTorch state
    - meta.json                   : id mappings, movie metadata, feature vocabs
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import load_movielens_1m
from src.models import ItemItemCosine
from src.models.hybrid import LightFMModel
from src.models.surprise_models import make_surprise_model

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "processed" / "artifacts"
ART.mkdir(parents=True, exist_ok=True)


def main() -> None:
    os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / ".surprise_data"))

    print("[1/5] Loading MovieLens 1M...")
    ml = load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
    train = ml.ratings  # use full dataset for production
    print(f"  {len(train):,} ratings")

    print("[2/5] Training ItemItemCosine...")
    cos = ItemItemCosine().fit(train)
    with open(ART / "cosine.pkl", "wb") as f:
        pickle.dump(cos, f)
    print(f"  saved {ART/'cosine.pkl'}")

    print("[3/5] Training Surprise SVD (50 factors, 20 epochs)...")
    svd = make_surprise_model("SVD", n_factors=50, n_epochs=20,
                              lr_all=0.005, reg_all=0.02, random_state=0).fit(train)
    with open(ART / "svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    print(f"  saved {ART/'svd.pkl'}")

    print("[4/5] Training LightFM hybrid (PyTorch, 32 factors, 8 epochs)...")
    lfm = LightFMModel(n_factors=32, n_epochs=8, batch_size=8192,
                       lr=0.005, reg=1e-5, loss="mse")
    lfm.build_user_features(ml.users)
    lfm.build_item_features(ml.movies)
    lfm.fit(train)
    state = {
        "model_state": lfm.model.state_dict(),
        "n_user_feats": len(lfm.user_feat_index_) + 1,
        "n_item_feats": len(lfm.item_feat_index_) + 1,
        "n_factors": lfm.n_factors,
    }
    torch = __import__("torch")
    torch.save(state, ART / "lightfm.pt")
    print(f"  saved {ART/'lightfm.pt'}")

    print("[5/5] Saving metadata...")
    movies_meta = ml.movies[["movie_id", "title", "year", "genres_list"]].copy()
    movies_meta["genres_str"] = movies_meta["genres_list"].apply(
        lambda g: "|".join(g) if isinstance(g, list) else ""
    )
    movies_meta = movies_meta.drop(columns=["genres_list"])
    movies_meta.to_csv(ART / "movies.csv", index=False)
    print(f"  saved {ART/'movies.csv'}")

    meta = {
        "n_users": int(ml.n_users),
        "n_items": int(ml.n_items),
        "n_ratings": int(ml.n_ratings),
        "user_feat_vocab": {f"{k[0]}={k[1]}": v for k, v in lfm.user_feat_index_.items()},
        "item_feat_vocab": {f"{k[0]}={k[1]}": v for k, v in lfm.item_feat_index_.items()},
    }
    with open(ART / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {ART/'meta.json'}")
    print("\nDone. Artifacts in:", ART)


if __name__ == "__main__":
    main()
