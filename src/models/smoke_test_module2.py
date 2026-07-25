"""Module 2 smoke test - full MovieLens 1M, fast models only."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (catalog_coverage, evaluate_ranking,
    evaluate_rating_prediction, novelty)
from src.models import ItemItemCosine, PopularityBaseline
from src.models.surprise_models import make_surprise_model

print("[1] Loading full MovieLens 1M...")
ml = load_movielens_1m(Path("data") / "raw" / "ml-1m")
print(f"    Ratings : {ml.n_ratings:,}")

train, test = temporal_split(ml.ratings, test_ratio=0.2)
print(f"Train: {len(train):,}  Test: {len(test):,}")

train_users = set(train.user_id.unique()); train_items = set(train.movie_id.unique())
test_eval = test[test.user_id.isin(train_users) & test.movie_id.isin(train_items)].copy()
rng = np.random.default_rng(0)
eval_users = list(rng.choice(test_eval.user_id.unique(), size=1000, replace=False))
test_subset = test_eval[test_eval.user_id.isin(eval_users)]

print("[2] Baselines...")
pop = PopularityBaseline(min_ratings=50, score="count").fit(train)
cos = ItemItemCosine().fit(train)

print("[3] Surprise models...")
models = {
    "BaselineOnly": make_surprise_model("BaselineOnly"),
    "SVD (50)": make_surprise_model("SVD", n_factors=50, n_epochs=20, random_state=0),
    "NMF (50)": make_surprise_model("NMF", n_factors=50, n_epochs=20, random_state=0),
}
fit_times = {}
for name, m in models.items():
    t0 = time.time()
    m.fit(train)
    fit_times[name] = time.time() - t0
    print(f"  {name:<20} fit {fit_times[name]:.1f}s")

print("\n[4] Rating prediction (50k pairs)...")
pairs = test_eval[["user_id","movie_id"]].drop_duplicates().head(50_000)
pairs = pairs.merge(test_eval[["user_id","movie_id","rating"]], on=["user_id","movie_id"], how="left").dropna()
for name, m in models.items():
    preds = m.predict_for_pairs(pairs[["user_id","movie_id"]])
    met = evaluate_rating_prediction(pairs, preds)
    print(f"  {name:<20} RMSE {met['rmse']:.4f} | MAE {met['mae']:.4f}")

print("\n[5] Top-10 ranking...")
all_recs = {"Popularity": pop.recommend_for_users(eval_users, top_k=50),
            "ItemItemCosine": cos.recommend_for_users(eval_users, top_k=50)}
for name, m in models.items():
    all_recs[name] = m.recommend_for_users(eval_users, top_k=50)
catalog = set(train_items)
item_pop = train.groupby("movie_id").size().to_dict()
rows = []
for name, recs in all_recs.items():
    r = evaluate_ranking(recs, test_subset, k=10)
    r["model"] = name; r["cov"] = catalog_coverage(recs, catalog, k=10)
    r["nov"] = novelty(recs, item_pop, k=10); r["fit_s"] = fit_times.get(name, 0)
    rows.append(r)
df = pd.DataFrame(rows).set_index("model")
df = df[["precision@10","recall@10","ndcg@10","hit_rate@10","map@10","cov","nov","fit_s"]]
print(df.round(4).to_string())
