"""Smoke test Module 4 - 100k sample, fast config."""
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (catalog_coverage, evaluate_ranking,
    evaluate_rating_prediction, novelty)
from src.models.neural_cf import TorchCF
from src.models.surprise_models import make_surprise_model

ml = load_movielens_1m(Path("data") / "raw" / "ml-1m")
ratings = ml.ratings.sample(n=100000, random_state=0).reset_index(drop=True)
print(f"Sample: {len(ratings):,}")
train, test = temporal_split(ratings, test_ratio=0.2)
train_sorted = train.sort_values("timestamp").reset_index(drop=True)
cut = int(len(train_sorted) * 0.9)
train_fit = train_sorted.iloc[:cut].copy(); val_fit = train_sorted.iloc[cut:].copy()

train_users = set(train.user_id.unique()); train_items = set(train.movie_id.unique())
test_eval = test[test.user_id.isin(train_users) & test.movie_id.isin(train_items)].copy()
rng = np.random.default_rng(0)
eval_users = list(rng.choice(test_eval.user_id.unique(), size=300, replace=False))
test_subset = test_eval[test_eval.user_id.isin(eval_users)]

print("TwoTower k=32 8 epochs...")
t0 = time.time()
tt = TorchCF(arch="twotower", n_factors=32, n_epochs=8, batch_size=4096,
             lr=0.005, reg=1e-5).fit(train_fit, val=val_fit)
print(f"  fit {time.time()-t0:.1f}s  val RMSE {tt.val_rmse_[-1]:.4f}")

print("NeuralCF hidden=(64,32) 8 epochs...")
t0 = time.time()
ncf = TorchCF(arch="neuralcf", n_factors=16, hidden=(64, 32), dropout=0.2,
              n_epochs=8, batch_size=4096, lr=0.001, reg=1e-4).fit(train_fit, val=val_fit)
print(f"  fit {time.time()-t0:.1f}s  val RMSE {ncf.val_rmse_[-1]:.4f}")

print("Surprise SVD (ref)...")
svd = make_surprise_model("SVD", n_factors=32, n_epochs=20, random_state=0).fit(train)

pairs = test_eval[["user_id","movie_id"]].drop_duplicates().head(10_000)
pairs = pairs.merge(test_eval[["user_id","movie_id","rating"]], on=["user_id","movie_id"], how="left").dropna()
print("\n--- Rating prediction ---")
rows = []
for name, m in [("TwoTower", tt), ("NeuralCF", ncf), ("Surprise SVD", svd)]:
    preds = m.predict_for_pairs(pairs[["user_id","movie_id"]])
    met = evaluate_rating_prediction(pairs, preds)
    met["model"] = name; rows.append(met)
df = pd.DataFrame(rows).set_index("model")[["rmse","mae","coverage"]]
print(df.round(4).to_string())

print("\n--- Top-10 ranking ---")
all_recs = {}
for name, m in [("TwoTower", tt), ("NeuralCF", ncf), ("Surprise SVD", svd)]:
    all_recs[name] = m.recommend_for_users(eval_users, top_k=50)
catalog = set(train_items); item_pop = train.groupby("movie_id").size().to_dict()
rows = []
for name, recs in all_recs.items():
    r = evaluate_ranking(recs, test_subset, k=10)
    r["model"] = name; r["cov"] = catalog_coverage(recs, catalog, k=10)
    r["nov"] = novelty(recs, item_pop, k=10); rows.append(r)
df = pd.DataFrame(rows).set_index("model")
df = df[["precision@10","recall@10","ndcg@10","hit_rate@10","map@10","cov","nov"]]
print(df.round(4).to_string())

curves_path = Path("data") / "processed" / "module4_curves.json"
curves_path.parent.mkdir(parents=True, exist_ok=True)
with curves_path.open("w") as f:
    json.dump({"twotower_train": tt.train_loss_, "twotower_val": tt.val_rmse_,
               "ncf_train": ncf.train_loss_, "ncf_val": ncf.val_rmse_}, f, indent=2)
print(f"Saved curves to {curves_path}")
