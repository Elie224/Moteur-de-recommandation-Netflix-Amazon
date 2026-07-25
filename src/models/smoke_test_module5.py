"""Smoke test Module 5 - Hybrid LightFM from scratch."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (catalog_coverage, evaluate_ranking,
    evaluate_rating_prediction, novelty)
from src.models.hybrid import LightFMModel
from src.models.surprise_models import make_surprise_model

print("[1] Loading MovieLens 1M...")
ml = load_movielens_1m(Path("data") / "raw" / "ml-1m")
ratings = ml.ratings.sample(n=100000, random_state=0).reset_index(drop=True)
print(f"Sample: {len(ratings):,}")
train, test = temporal_split(ratings, test_ratio=0.2)

# Filter users/movies to those in train
train_users = set(train.user_id.unique()); train_items = set(train.movie_id.unique())
ml_users = ml.users[ml.users.user_id.isin(train_users)]
ml_movies = ml.movies[ml.movies.movie_id.isin(train_items)]

test_eval = test[test.user_id.isin(train_users) & test.movie_id.isin(train_items)].copy()
rng = np.random.default_rng(0)
eval_users = list(rng.choice(test_eval.user_id.unique(), size=300, replace=False))
test_subset = test_eval[test_eval.user_id.isin(eval_users)]

print("[2] LightFM (MSE, k=32)...")
t0 = time.time()
lfm = LightFMModel(n_factors=32, n_epochs=8, batch_size=4096, lr=0.005, reg=1e-5, loss="mse")
lfm.build_user_features(ml_users)
lfm.build_item_features(ml_movies)
print(f"  user features: {len(lfm.user_feat_index_)}, item features: {len(lfm.item_feat_index_)}")
lfm.fit(train)
print(f"  fit {time.time()-t0:.1f}s  final RMSE {lfm.train_loss_[-1]:.4f}")

print("[3] Surprise SVD (reference)...")
svd = make_surprise_model("SVD", n_factors=32, n_epochs=20, random_state=0).fit(train)

# Rating prediction
pairs = test_eval[["user_id","movie_id"]].drop_duplicates().head(10_000)
pairs = pairs.merge(test_eval[["user_id","movie_id","rating"]], on=["user_id","movie_id"], how="left").dropna()
print("\n--- Rating prediction ---")
rows = []
for name, m in [("LightFM (hybrid, ours)", lfm), ("Surprise SVD", svd)]:
    preds = m.predict_for_pairs(pairs[["user_id","movie_id"]])
    met = evaluate_rating_prediction(pairs, preds)
    met["model"] = name; rows.append(met)
df = pd.DataFrame(rows).set_index("model")[["rmse","mae","coverage"]]
print(df.round(4).to_string())

print("\n--- Top-10 ranking ---")
all_recs = {"LightFM (hybrid, ours)": lfm.recommend_for_users(eval_users, top_k=50),
            "Surprise SVD": svd.recommend_for_users(eval_users, top_k=50)}
catalog = set(train_items); item_pop = train.groupby("movie_id").size().to_dict()
rows = []
for name, recs in all_recs.items():
    r = evaluate_ranking(recs, test_subset, k=10)
    r["model"] = name; r["cov"] = catalog_coverage(recs, catalog, k=10)
    r["nov"] = novelty(recs, item_pop, k=10); rows.append(r)
df = pd.DataFrame(rows).set_index("model")
df = df[["precision@10","recall@10","ndcg@10","hit_rate@10","map@10","cov","nov"]]
print(df.round(4).to_string())

print("\n--- Cold-start demo: recommend for a brand-new user (no history) ---")
new_user_features = [("gender", "F"), ("age", "25"), ("occupation", "10")]
recs = lfm.recommend_for_new_user(new_user_features, top_k=10)
print(f"New user {new_user_features}:")
for iid, score in recs[:5]:
    print(f"  {score:+.3f}  {ml.movies.set_index('movie_id').loc[iid, 'title']}")

curves_path = Path("data") / "processed" / "module5_curves.json"
curves_path.parent.mkdir(parents=True, exist_ok=True)
with curves_path.open("w") as f:
    json.dump({"lightfm_rmse": lfm.train_loss_}, f, indent=2)
print(f"Saved curves to {curves_path}")
