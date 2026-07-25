"""Smoke test Module 3 - FunkSVD and ALS from scratch vs Surprise SVD."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (catalog_coverage, evaluate_ranking,
    evaluate_rating_prediction, novelty)
from src.models import ItemItemCosine, PopularityBaseline
from src.models.surprise_models import make_surprise_model
from src.models.matrix_factorization import FunkSVD, ALSMF

print("[1] Loading MovieLens 1M...")
ml = load_movielens_1m(Path("data") / "raw" / "ml-1m")

train, test = temporal_split(ml.ratings, test_ratio=0.2)
train_users = set(train.user_id.unique()); train_items = set(train.movie_id.unique())
test_eval = test[test.user_id.isin(train_users) & test.movie_id.isin(train_items)].copy()
rng = np.random.default_rng(0)
eval_users = list(rng.choice(test_eval.user_id.unique(), size=500, replace=False))
test_subset = test_eval[test_eval.user_id.isin(eval_users)]

# Validation split from train (last 10% of train) for FunkSVD training curve
train_sorted = train.sort_values("timestamp").reset_index(drop=True)
cut = int(len(train_sorted) * 0.9)
train_fit = train_sorted.iloc[:cut].copy()
val_fit = train_sorted.iloc[cut:].copy()
print(f"  train_fit: {len(train_fit):,}  val_fit: {len(val_fit):,}")

print("\n[2] Fitting FunkSVD (50 factors, 20 epochs)...")
t0 = time.time()
funk = FunkSVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02, reg_bias=0.02,
               batch_size=8192, random_state=0, verbose=False)
funk.fit(train, val=val_fit)
print(f"  fit in {time.time()-t0:.1f}s")
print(f"  final train RMSE: {funk.train_loss_[-1]:.4f}")
print(f"  final val   RMSE: {funk.val_rmse_[-1]:.4f}")

print("\n[3] Fitting Surprise SVD (same config)...")
t0 = time.time()
svd = make_surprise_model("SVD", n_factors=50, n_epochs=20, lr_all=0.005,
                          reg_all=0.02, random_state=0).fit(train)
print(f"  fit in {time.time()-t0:.1f}s")

print("\n[4] Fitting ALS-MF (15 epochs)...")
t0 = time.time()
als = ALSMF(n_factors=50, n_epochs=15, reg=0.05, reg_bias=0.02).fit(train)
print(f"  fit in {time.time()-t0:.1f}s")
print(f"  final train RMSE: {als.train_rmse_[-1]:.4f}")

# Rating prediction
print("\n[5] Rating prediction (50k pairs)...")
pairs = test_eval[["user_id","movie_id"]].drop_duplicates().head(50_000)
pairs = pairs.merge(test_eval[["user_id","movie_id","rating"]], on=["user_id","movie_id"], how="left").dropna()
for name, m in [("FunkSVD (ours)", funk), ("Surprise SVD", svd), ("ALS-MF (ours)", als)]:
    preds = m.predict_for_pairs(pairs[["user_id","movie_id"]])
    met = evaluate_rating_prediction(pairs, preds)
    print(f"  {name:<20} RMSE {met['rmse']:.4f} | MAE {met['mae']:.4f} | cov {met['coverage']:.2%}")

# Ranking
print("\n[6] Top-10 ranking (500 users)...")
all_recs = {}
for name, m in [("FunkSVD (ours)", funk), ("Surprise SVD", svd), ("ALS-MF (ours)", als)]:
    t0 = time.time()
    all_recs[name] = m.recommend_for_users(eval_users, top_k=50)
    print(f"  {name:<20} recs {time.time()-t0:.1f}s")
catalog = set(train_items)
item_pop = train.groupby("movie_id").size().to_dict()
rows = []
for name, recs in all_recs.items():
    r = evaluate_ranking(recs, test_subset, k=10)
    r["model"] = name; r["cov"] = catalog_coverage(recs, catalog, k=10)
    r["nov"] = novelty(recs, item_pop, k=10)
    rows.append(r)
df = pd.DataFrame(rows).set_index("model")
df = df[["precision@10","recall@10","ndcg@10","hit_rate@10","map@10","cov","nov"]]
print(df.round(4).to_string())

# Save training curves for the notebook
import json
curves_path = Path("data") / "processed" / "module3_curves.json"
curves_path.parent.mkdir(parents=True, exist_ok=True)
curves = {
    "funksvd_train_rmse": funk.train_loss_,
    "funksvd_val_rmse": funk.val_rmse_,
    "als_train_rmse": als.train_rmse_,
}
with curves_path.open("w") as f:
    json.dump(curves, f, indent=2)
print(f"\nSaved training curves to {curves_path}")
