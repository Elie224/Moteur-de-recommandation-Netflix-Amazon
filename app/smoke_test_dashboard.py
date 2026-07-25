"""Smoke test for the Streamlit dashboard module loaders."""
import os, sys, json, pickle
from pathlib import Path

os.environ.setdefault("SURPRISE_DATA_FOLDER", str(Path.cwd() / ".surprise_data"))
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd, numpy as np, torch
ART = ROOT / "data" / "processed" / "artifacts"

with open(ART / "cosine.pkl", "rb") as f:
    cos = pickle.load(f)
with open(ART / "svd.pkl", "rb") as f:
    svd = pickle.load(f)
lfm_state = torch.load(ART / "lightfm.pt", map_location="cpu", weights_only=False)
movies = pd.read_csv(ART / "movies.csv")
with open(ART / "meta.json") as f:
    meta = json.load(f)

print("Models loaded")
print("  cosine shape:", cos.item_similarity_.shape)
print("  svd type:", type(svd.algo).__name__)
print("  lfm_state keys:", len(lfm_state["model_state"]))
print("  movies:", len(movies))

print("\nRecommendations for user 1 (cosine):")
for iid, score in cos.recommend(1, top_k=5):
    title = movies.set_index("movie_id").loc[iid, "title"] if iid in movies.movie_id.values else str(iid)
    print(f"  {score:+.3f}  {title}")

print("\nRecommendations for user 1 (svd):")
for iid, score in svd.recommend(1, top_k=5):
    title = movies.set_index("movie_id").loc[iid, "title"] if iid in movies.movie_id.values else str(iid)
    print(f"  {score:+.3f}  {title}")

pair = pd.DataFrame({"user_id": [1], "movie_id": [1]})
rating_cos = list(cos.predict_for_pairs(pair).values())[0]
rating_svd = list(svd.predict_for_pairs(pair).values())[0]
print(f"\nPredict (1, 1) -> cosine={rating_cos:.3f}  svd={rating_svd:.3f}")

print("\nCold-start LightFM build...")
from src.models.hybrid import LightFMModel, HybridCF
from src.data.loaders import load_movielens_1m
ml = load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
lfm = LightFMModel(n_factors=lfm_state["n_factors"])
lfm.user_feat_index_ = {(k.split("=", 1)[0], k.split("=", 1)[1]): v for k, v in meta["user_feat_vocab"].items()}
lfm.item_feat_index_ = {(k.split("=", 1)[0], k.split("=", 1)[1]): v for k, v in meta["item_feat_vocab"].items()}
ru = np.sort(ml.ratings["user_id"].unique())
ri = np.sort(ml.ratings["movie_id"].unique())
lfm.user_index_ = {int(u): i for i, u in enumerate(ru)}
lfm.item_index_ = {int(m): i for i, m in enumerate(ri)}
users_df = ml.users.set_index("user_id").loc[ru].reset_index()
movies_df = ml.movies.set_index("movie_id").loc[ri].reset_index()
uk = [[("gender", str(r["gender"])), ("age", str(r["age"])), ("occupation", str(r["occupation"]))] for _, r in users_df.iterrows()]
max_u = max(len(k) for k in uk)
u_mat = np.zeros((len(ru), max_u), dtype=np.int64)
for i, ks in enumerate(uk):
    for j, k in enumerate(ks):
        if k in lfm.user_feat_index_:
            u_mat[i, j] = lfm.user_feat_index_[k]
lfm.user_features_mat_ = torch.tensor(u_mat)
ik = []
for _, row in movies_df.iterrows():
    ks = [("genre", g) for g in (row["genres_list"] or [])]
    try:
        year = int(row["year"]) if row["year"] == row["year"] else None
    except Exception:
        year = None
    if year:
        ks.append(("decade", str((year // 10) * 10)))
    ik.append(ks)
max_i = max(len(k) for k in ik) if ik else 1
i_mat = np.zeros((len(ri), max_i), dtype=np.int64)
for i, ks in enumerate(ik):
    for j, k in enumerate(ks):
        if k in lfm.item_feat_index_:
            i_mat[i, j] = lfm.item_feat_index_[k]
lfm.item_features_mat_ = torch.tensor(i_mat)
lfm.model = HybridCF(n_users=len(lfm.user_index_), n_items=len(lfm.item_index_),
                     n_user_features=lfm_state["n_user_feats"], n_item_features=lfm_state["n_item_feats"],
                     n_factors=lfm_state["n_factors"])
lfm.model.load_state_dict(lfm_state["model_state"])
lfm.model.eval()
print("LightFM built OK")

print("\nCold-start recs for (F, 25, occ=10):")
for iid, score in lfm.recommend_for_new_user([("gender", "F"), ("age", "25"), ("occupation", "10")], top_k=5):
    title = movies.set_index("movie_id").loc[iid, "title"] if iid in movies.movie_id.values else str(iid)
    print(f"  {score:+.3f}  {title}")
print("\nAll dashboard components working.")
