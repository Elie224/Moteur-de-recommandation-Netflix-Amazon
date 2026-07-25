"""Smoke test for Module 1.

Run from the repo root:
    .venv/Scripts/python src/models/smoke_test_module1.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# Make ``src`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (
    catalog_coverage,
    evaluate_ranking,
    evaluate_rating_prediction,
    novelty,
)
from src.models import ItemItemCosine, PopularityBaseline


def main() -> None:
    root = Path("data") / "raw" / "ml-1m"
    print("[1] Loading MovieLens 1M...")
    t0 = time.time()
    ml = load_movielens_1m(root)
    print(f"    Ratings : {ml.n_ratings:,}")
    print(f"    Users   : {ml.n_users:,}")
    print(f"    Items   : {ml.n_items:,}")
    print(f"    Sparsity: {ml.sparsity:.4%}")
    print(f"    Loaded in {time.time() - t0:.1f}s")

    print("[2] Temporal split (80/20)...")
    train, test = temporal_split(ml.ratings, test_ratio=0.2)
    print(f"    Train: {len(train):,}  Test: {len(test):,}")

    train_users = set(train["user_id"].unique())
    train_items = set(train["movie_id"].unique())
    test_eval = test[
        test["user_id"].isin(train_users) & test["movie_id"].isin(train_items)
    ].copy()
    print(f"    Eval rows (no cold): {len(test_eval):,}")

    print("[3] Fitting popularity baseline...")
    pop = PopularityBaseline(min_ratings=50, score="count").fit(train)

    print("[4] Fitting item-item cosine baseline...")
    t0 = time.time()
    cos = ItemItemCosine().fit(train)
    print(f"    Fit in {time.time() - t0:.1f}s")
    print(f"    Item-item similarity matrix: {cos.item_similarity_.shape}")

    sample_user = int(test_eval["user_id"].iloc[0])
    print(f"\n[5] Top-10 recommendations for user {sample_user}:")
    title_map = dict(zip(ml.movies["movie_id"], ml.movies["title"]))
    seen = set(train[train["user_id"] == sample_user]["movie_id"].tolist())
    recs = cos.recommend(sample_user, top_k=10, seen_items=seen)
    for rank, (iid, score) in enumerate(recs, 1):
        print(f"   {rank:2d}. [{score:+.4f}] {title_map.get(iid, iid)}")

    print("\n[6] Rating prediction (RMSE / MAE)...")
    pairs = test_eval[["user_id", "movie_id"]].drop_duplicates().head(50_000)
    pairs_with_rating = pairs.merge(
        test_eval[["user_id", "movie_id", "rating"]],
        on=["user_id", "movie_id"],
        how="left",
    ).dropna()
    cos_preds = cos.predict_for_pairs(pairs_with_rating[["user_id", "movie_id"]])
    metrics = evaluate_rating_prediction(pairs_with_rating, cos_preds)
    print(f"    ItemItemCosine  -> RMSE {metrics['rmse']:.4f} | MAE {metrics['mae']:.4f} | coverage {metrics['coverage']:.2%}")

    print("\n[7] Top-10 ranking metrics...")
    eval_users = test_eval["user_id"].unique().tolist()
    rng = np.random.default_rng(0)
    eval_users = list(rng.choice(eval_users, size=min(1000, len(eval_users)), replace=False))
    test_subset = test_eval[test_eval["user_id"].isin(eval_users)]

    t0 = time.time()
    cos_recs = cos.recommend_for_users(eval_users, top_k=50)
    pop_recs = pop.recommend_for_users(eval_users, top_k=50)
    print(f"    Generated recs in {time.time() - t0:.1f}s")

    cos_metrics = evaluate_ranking(cos_recs, test_subset, k=10)
    pop_metrics = evaluate_ranking(pop_recs, test_subset, k=10)
    print(f"    Popularity       -> P@10 {pop_metrics['precision@10']:.4f} | R@10 {pop_metrics['recall@10']:.4f} | NDCG@10 {pop_metrics['ndcg@10']:.4f} | HR@10 {pop_metrics['hit_rate@10']:.4f} | n_users {pop_metrics['n_users_evaluated']}")
    print(f"    ItemItemCosine   -> P@10 {cos_metrics['precision@10']:.4f} | R@10 {cos_metrics['recall@10']:.4f} | NDCG@10 {cos_metrics['ndcg@10']:.4f} | HR@10 {cos_metrics['hit_rate@10']:.4f} | n_users {cos_metrics['n_users_evaluated']}")

    catalog = set(train_items)
    cov_pop = catalog_coverage(pop_recs, catalog, k=10)
    cov_cos = catalog_coverage(cos_recs, catalog, k=10)
    item_pop = train.groupby("movie_id").size().to_dict()
    nov_pop = novelty(pop_recs, item_pop, k=10)
    nov_cos = novelty(cos_recs, item_pop, k=10)
    print("\n[8] Beyond accuracy:")
    print(f"    Popularity       -> catalog coverage {cov_pop:.2%} | novelty {nov_pop:.3f}")
    print(f"    ItemItemCosine   -> catalog coverage {cov_cos:.2%} | novelty {nov_cos:.3f}")


if __name__ == "__main__":
    np.random.seed(42)
    main()

