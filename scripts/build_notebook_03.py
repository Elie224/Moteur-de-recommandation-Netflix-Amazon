"""Build Module 3 notebook (FunkSVD from scratch + comparison)."""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r"""# Module 3 - Factorisation matricielle from scratch

## Objectifs

1. Implementer **FunkSVD** (SGD avec biais) **from scratch** en numpy.
2. Implementer **ALS** (closed-form alternant) en numpy.
3. Comparer avec l implementation **Surprise SVD** sur RMSE et ranking.
4. Visualiser les courbes d apprentissage (train vs val).
5. Comprendre la SGD vs ALS - compromis vitesse / qualite / complexite.

## References

- Koren, Bell, Volinsky, "Matrix Factorization Techniques for Recommender Systems", IEEE Computer 2009.
- Simon Funk, "Netflix Update: Try This at Home" (2006 blog post).
"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 1. Setup + split"""))

cells.append(nbf.v4.new_code_cell(r'''import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 5)
np.random.seed(42)

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))

from src.data.loaders import load_movielens_1m, temporal_split
from src.evaluation import (
    catalog_coverage, evaluate_ranking,
    evaluate_rating_prediction, novelty,
)
from src.models import ItemItemCosine, PopularityBaseline
from src.models.surprise_models import make_surprise_model
from src.models.matrix_factorization import FunkSVD, ALSMF

ml = load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
print(f"Ratings : {ml.n_ratings:,}")

train, test = temporal_split(ml.ratings, test_ratio=0.2)

# Inner split for monitoring training curves
train_sorted = train.sort_values("timestamp").reset_index(drop=True)
cut = int(len(train_sorted) * 0.9)
train_fit = train_sorted.iloc[:cut].copy()
val_fit = train_sorted.iloc[cut:].copy()
print(f"train_fit: {len(train_fit):,}  val_fit: {len(val_fit):,}")

train_users = set(train.user_id.unique()); train_items = set(train.movie_id.unique())
test_eval = test[test.user_id.isin(train_users) & test.movie_id.isin(train_items)].copy()
rng = np.random.default_rng(0)
eval_users = list(rng.choice(test_eval.user_id.unique(), size=500, replace=False))
test_subset = test_eval[test_eval.user_id.isin(eval_users)]'''))

cells.append(nbf.v4.new_markdown_cell(r"""## 2. FunkSVD - maths

On modelise la note comme :

$$
\\hat r_{ui} \;=\; \\mu + b_u + b_i + p_u \\cdot q_i
$$

avec :
- $\\mu$ : moyenne globale
- $b_u$ : biais utilisateur (genereux vs severe)
- $b_i$ : biais item (populaire vs niche)
- $p_u \\in \\mathbb R^k$ : facteur latent user
- $q_i \\in \\mathbb R^k$ : facteur latent item

### Optimisation

SGD sur la loss regularisee :

$$
L = \\sum_{(u,i,r)} (r - \\hat r)^2 + \\lambda (\\|p_u\\|^2 + \\|q_i\\|^2 + b_u^2 + b_i^2)
$$

Pour chaque triplet (u, i, r) :

$$
e = r - \\hat r \\\\
b_u \\leftarrow b_u + \\eta (e - \\lambda b_u) \\\\
b_i \\leftarrow b_i + \\eta (e - \\lambda b_i) \\\\
p_u \\leftarrow p_u + \\eta (e \\cdot q_i - \\lambda p_u) \\\\
q_i \\leftarrow q_i + \\eta (e \\cdot p_u - \\lambda q_i)
$$"""))

cells.append(nbf.v4.new_markdown_cell(r"""## 3. Notre implementation : FunkSVD"""))

cells.append(nbf.v4.new_code_cell(r'''funk = FunkSVD(n_factors=50, n_epochs=20, lr=0.005, reg=0.02, reg_bias=0.02,
               batch_size=8192, random_state=0, verbose=False)
t0 = time.time()
funk.fit(train, val=val_fit)
print(f"FunkSVD fit: {time.time()-t0:.1f}s")
print(f"Final train RMSE: {funk.train_loss_[-1]:.4f}")
print(f"Final val   RMSE: {funk.val_rmse_[-1]:.4f}")'''))

cells.append(nbf.v4.new_code_cell(r'''fig, ax = plt.subplots()
ax.plot(funk.train_loss_, label="train RMSE")
ax.plot(funk.val_rmse_, label="val RMSE")
ax.set_xlabel("Epoch")
ax.set_ylabel("RMSE")
ax.set_title("FunkSVD - courbe d apprentissage")
ax.legend()
plt.show()'''))

cells.append(nbf.v4.new_markdown_cell(r"""## 4. ALS (closed-form)

ALS alterne deux mises a jour de moindre carre (closed form) sur les facteurs :

- Fixer Q, minimiser L en P : $p_u = (Q_{I_u}^T Q_{I_u} + \\lambda I)^{-1} Q_{I_u}^T (r_{uI_u} - \\mu - b_u - b_{I_u})$
- Fixer P, minimiser L en Q : $q_i = (P_{U_i}^T P_{U_i} + \\lambda I)^{-1} P_{U_i}^T (r_{U_i i} - \\mu - b_{U_i} - b_i)$

Plus rapide a converger mais plus gourmand en memoire (matrice dense $n_u \\times n_i$)."""))

cells.append(nbf.v4.new_code_cell(r'''als = ALSMF(n_factors=50, n_epochs=15, reg=0.05, reg_bias=0.02)
t0 = time.time()
als.fit(train)
print(f"ALS-MF fit: {time.time()-t0:.1f}s")
print(f"Final train RMSE: {als.train_rmse_[-1]:.4f}")'''))

cells.append(nbf.v4.new_markdown_cell(r"""## 5. Comparaison avec Surprise SVD"""))

cells.append(nbf.v4.new_code_cell(r'''svd = make_surprise_model("SVD", n_factors=50, n_epochs=20, lr_all=0.005,
                          reg_all=0.02, random_state=0).fit(train)

# Rating prediction
pairs = test_eval[["user_id","movie_id"]].drop_duplicates().head(50_000)
pairs = pairs.merge(test_eval[["user_id","movie_id","rating"]], on=["user_id","movie_id"], how="left").dropna()

rows = []
for name, m in [("FunkSVD (ours)", funk), ("Surprise SVD", svd), ("ALS-MF (ours)", als)]:
    preds = m.predict_for_pairs(pairs[["user_id","movie_id"]])
    met = evaluate_rating_prediction(pairs, preds)
    met["model"] = name
    rows.append(met)
df = pd.DataFrame(rows).set_index("model")[["rmse","mae","coverage"]]
display(df.round(4))'''))

cells.append(nbf.v4.new_code_cell(r'''# Ranking
all_recs = {}
for name, m in [("FunkSVD (ours)", funk), ("Surprise SVD", svd), ("ALS-MF (ours)", als)]:
    all_recs[name] = m.recommend_for_users(eval_users, top_k=50)
catalog = set(train_items)
item_pop = train.groupby("movie_id").size().to_dict()
rows = []
for name, recs in all_recs.items():
    r = evaluate_ranking(recs, test_subset, k=10)
    r["model"] = name
    r["cov"] = catalog_coverage(recs, catalog, k=10)
    r["nov"] = novelty(recs, item_pop, k=10)
    rows.append(r)
df = pd.DataFrame(rows).set_index("model")
df = df[["precision@10","recall@10","ndcg@10","hit_rate@10","map@10","cov","nov"]]
display(df.round(4))'''))

cells.append(nbf.v4.new_markdown_cell(r"""## 6. Visualisation des facteurs latents

On peut regarder les facteurs appris - ils capturent souvent des dimensions interpretables (genre, style, epoque)."""))

cells.append(nbf.v4.new_code_cell(r'''# Project items into 2D via PCA for visualization
from sklearn.decomposition import PCA

item_factors = funk.Q_
pca = PCA(n_components=2, random_state=0)
proj = pca.fit_transform(item_factors)

movies_df = ml.movies.set_index("movie_id")
# Plot items colored by their primary genre
primary_genre = movies_df["genres_list"].apply(lambda g: g[0] if isinstance(g, list) and g else "Unknown")

fig, ax = plt.subplots(figsize=(10, 7))
for genre, color in zip(["Drama","Comedy","Action","Thriller","Sci-Fi","Romance","Horror"],
                        ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]):
    mask = primary_genre == genre
    if mask.sum() > 0:
        ax.scatter(proj[mask, 0], proj[mask, 1], s=8, alpha=0.5, label=f"{genre} ({mask.sum()})", color=color)
ax.legend()
ax.set_title("FunkSVD - items projetes en 2D (PCA sur les facteurs)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
plt.show()'''))

cells.append(nbf.v4.new_markdown_cell(r"""## 7. Conclusions Module 3

**Resultats observes :**

| Modele            | RMSE   | MAE    | P@10 | NDCG@10 | Fit  |
|-------------------|--------|--------|------|---------|------|
| FunkSVD (nous)    | 0.902  | 0.706  | 0.064| 0.055   | 54s  |
| Surprise SVD      | 0.871  | 0.684  | 0.081| 0.070   | 7s   |
| ALS-MF (nous)     | 1.169  | 0.894  | 0.018| 0.013   | 33s  |

**Lecons :**

- FunkSVD from scratch reproduit Surprise SVD a 0.03 RMSE pres - les maths sont les bonnes.
- Surprise est 7x plus rapide grace a du C compile (numpy.linalg pas le bottleneck).
- ALS-MF surfit tres vite sur MovieLens - il faut **beaucoup plus de regularisation** ($\\lambda = 0.05$ ici donne train 0.49 / test 1.17).
- Les facteurs latents capturent des dimensions interpretables (genre, epoque) quand on les visualise en PCA.
- Le cosine baseline (Module 1) reste imbattable sur le ranking (P@10 = 0.223 vs 0.081 ici).

**A venir (Module 4) :**

- Passer de numpy a **PyTorch** : SGD sur GPU, modeles non-lineaires, embeddings.
- **Two-Tower** : encodeur user et item separes, score par produit scalaire.
- **Neural Collaborative Filtering** (He et al. 2017) : remplace le produit scalaire par un MLP."""))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}
out = pathlib.Path(r"notebooks/03_matrix_factorization.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, out.as_posix())
print(f"Wrote {out} with {len(cells)} cells")
