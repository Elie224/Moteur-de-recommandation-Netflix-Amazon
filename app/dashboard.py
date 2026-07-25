"""Streamlit dashboard for the movie recommendation engine.

Run:
    .venv/Scripts/python -m streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SURPRISE_DATA_FOLDER", str(ROOT / ".surprise_data"))

ART = ROOT / "data" / "processed" / "artifacts"


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------


@st.cache_resource
def load_models():
    with open(ART / "cosine.pkl", "rb") as f:
        cos = pickle.load(f)
    with open(ART / "svd.pkl", "rb") as f:
        svd = pickle.load(f)
    lfm_state = torch.load(ART / "lightfm.pt", map_location="cpu", weights_only=False)
    movies = pd.read_csv(ART / "movies.csv")
    with open(ART / "meta.json") as f:
        meta = json.load(f)
    return cos, svd, lfm_state, movies, meta


@st.cache_resource
def build_lightfm(_meta: dict, _lfm_state: dict):
    from src.data.loaders import load_movielens_1m
    from src.models.hybrid import LightFMModel, HybridCF

    ml = load_movielens_1m(ROOT / "data" / "raw" / "ml-1m")
    lfm = LightFMModel(n_factors=_lfm_state["n_factors"])
    lfm.user_feat_index_ = {
        (k.split("=", 1)[0], k.split("=", 1)[1]): v
        for k, v in _meta["user_feat_vocab"].items()
    }
    lfm.item_feat_index_ = {
        (k.split("=", 1)[0], k.split("=", 1)[1]): v
        for k, v in _meta["item_feat_vocab"].items()
    }
    ru = np.sort(ml.ratings["user_id"].unique())
    ri = np.sort(ml.ratings["movie_id"].unique())
    lfm.user_index_ = {int(u): i for i, u in enumerate(ru)}
    lfm.item_index_ = {int(m): i for i, m in enumerate(ri)}
    users_df = ml.users.set_index("user_id").loc[ru].reset_index()
    movies_df = ml.movies.set_index("movie_id").loc[ri].reset_index()
    uk = [[("gender", str(r["gender"])), ("age", str(r["age"])), ("occupation", str(r["occupation"]))]
          for _, r in users_df.iterrows()]
    max_u = max(len(ks) for ks in uk)
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
        except (TypeError, ValueError):
            year = None
        if year:
            ks.append(("decade", str((year // 10) * 10)))
        ik.append(ks)
    max_i = max(len(ks) for ks in ik) if ik else 1
    i_mat = np.zeros((len(ri), max_i), dtype=np.int64)
    for i, ks in enumerate(ik):
        for j, k in enumerate(ks):
            if k in lfm.item_feat_index_:
                i_mat[i, j] = lfm.item_feat_index_[k]
    lfm.item_features_mat_ = torch.tensor(i_mat)
    lfm.model = HybridCF(
        n_users=len(lfm.user_index_), n_items=len(lfm.item_index_),
        n_user_features=_lfm_state["n_user_feats"],
        n_item_features=_lfm_state["n_item_feats"],
        n_factors=_lfm_state["n_factors"],
    )
    lfm.model.load_state_dict(_lfm_state["model_state"])
    lfm.model.eval()
    return lfm, movies_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def stars(rating: float, max_stars: int = 5) -> str:
    """Render a rating as filled/empty stars."""
    full = int(round(rating))
    full = max(0, min(max_stars, full))
    return ":star:" * full + ":material/star_border:" * (max_stars - full)


def render_movie_card(row: pd.Series, score: float | None = None,
                      rating: float | None = None) -> None:
    """Render a single recommendation as a card."""
    title = row.get("title", "?")
    year = row.get("year")
    year_str = f" ({int(year)})" if year == year else ""
    genres = row.get("genres_str", "")
    with st.container(border=True):
        sub = st.columns([3, 1])
        with sub[0]:
            st.markdown(f"**{title}**{year_str}")
            st.caption(genres if isinstance(genres, str) and genres else "No genre")
        with sub[1]:
            if rating is not None:
                st.markdown(f"{stars(rating)}  **{rating:.2f}**")
            elif score is not None:
                st.metric("Score", f"{float(score):+.3f}")


def get_genres_list(movies: pd.DataFrame) -> list[str]:
    """Aggregate unique genres from movies metadata."""
    all_g = set()
    for gs in movies["genres_str"].dropna():
        for g in str(gs).split("|"):
            if g:
                all_g.add(g)
    return sorted(all_g)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------


st.set_page_config(
    page_title="Moteur de recommandation",
    layout="wide",
    page_icon=":material/movie_filter:",
)

cos, svd, lfm_state, movies, meta = load_models()
all_genres = get_genres_list(movies)

# Top header
st.title(":material/movie_filter: Moteur de recommandation Netflix / Amazon")
st.caption("MovieLens 1M, collaborative filtering, matrix factorization, neural embeddings, hybride LightFM.")

# Sidebar: dataset stats + filters
with st.sidebar:
    st.header(":material/insights: Dataset")
    st.metric("Utilisateurs", f"{meta['n_users']:,}")
    st.metric("Films", f"{meta['n_items']:,}")
    st.metric("Notes", f"{meta['n_ratings']:,}")
    with st.container(border=True):
        st.markdown("**Features**")
        st.write(f"User features: {len(meta['user_feat_vocab'])}")
        st.write(f"Item features: {len(meta['item_feat_vocab'])}")
    st.markdown("---")
    st.markdown("**API FastAPI** : [localhost:8000/docs](http://localhost:8000/docs)")
    st.markdown("**Code source** : `notebooks/01..05`")

# Tabs
tab_rec, tab_predict, tab_cold, tab_compare = st.tabs([
    ":material/recommend: Recommander",
    ":material/calculate: Predire une note",
    ":material/person_add: Cold start",
    ":material/bar_chart: Comparer",
])

# ---------------------------------------------------------------------------
# Tab 1: Recommend
# ---------------------------------------------------------------------------
with tab_rec:
    st.header("Top-K pour un utilisateur")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
        user_id = c1.number_input("User ID", min_value=1, max_value=meta["n_users"], value=1, step=1)
        model_name = c2.segmented_control(
            "Modele",
            options=["cosine", "svd"],
            default="cosine",
        )
        top_k = c3.slider("Top-K", min_value=5, max_value=50, value=10)
        genre_filter = c4.pills("Filtrer genre", all_genres, selection_mode="multi")

    if st.button("Generer les recommandations", type="primary", icon=":material/play_arrow:"):
        m = cos if model_name == "cosine" else svd
        recs = m.recommend(int(user_id), top_k=max(top_k, 50))
        if not recs:
            st.warning("Aucune recommandation pour cet utilisateur (cold user).")
        else:
            filtered = []
            for iid, score in recs:
                row = movies[movies.movie_id == iid]
                if row.empty:
                    continue
                row = row.iloc[0]
                if genre_filter:
                    gset = set(str(row["genres_str"]).split("|"))
                    if not (set(genre_filter) & gset):
                        continue
                filtered.append((row, score))
                if len(filtered) >= top_k:
                    break
            if not filtered:
                st.warning("Aucun film ne correspond aux filtres.")
            else:
                st.subheader(f"Recommandations pour user {user_id} ({model_name})")
                # Two-column layout for cards
                cols = st.columns(2)
                for i, (row, score) in enumerate(filtered):
                    with cols[i % 2]:
                        render_movie_card(row, score=score)

# ---------------------------------------------------------------------------
# Tab 2: Predict a rating
# ---------------------------------------------------------------------------
with tab_predict:
    st.header("Predire la note pour (user, movie)")
    with st.form("predict_form"):
        c1, c2 = st.columns([1, 3])
        with c1:
            uid_p = st.number_input("User ID", min_value=1, max_value=meta["n_users"],
                                    value=1, step=1)
        with c2:
            query = st.text_input("Recherche film", placeholder="Tapez au moins 1 caractere...")
            if query:
                matches = movies[movies.title.str.contains(query, case=False, na=False)].head(20)
            else:
                matches = movies.head(20)
            if not matches.empty:
                options = [f"{int(r.movie_id)} - {r.title}" for _, r in matches.iterrows()]
                choice = st.selectbox("Film", options)
                movie_id = int(choice.split(" - ")[0])
            else:
                st.info("Aucun film trouve.")
                movie_id = 1
        submitted = st.form_submit_button("Predire", type="primary", icon=":material/search:")

    if submitted:
        pair = pd.DataFrame({"user_id": [int(uid_p)], "movie_id": [int(movie_id)]})
        rating_cos = list(cos.predict_for_pairs(pair).values())[0]
        rating_svd = list(svd.predict_for_pairs(pair).values())[0]
        st.subheader("Resultats")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            with st.container(border=True):
                st.markdown("**ItemItemCosine**")
                st.markdown(f"### {rating_cos:.2f} / 5")
                st.progress(min(rating_cos / 5, 1.0))
                st.caption("KNN-with-means, item-item")
        with c2:
            with st.container(border=True):
                st.markdown("**Surprise SVD**")
                st.markdown(f"### {rating_svd:.2f} / 5")
                st.progress(min(rating_svd / 5, 1.0))
                st.caption("Factorisation matricielle 50 facteurs")
        with c3:
            with st.container(border=True):
                delta = rating_svd - rating_cos
                st.markdown("**Ecart (SVD - Cosine)**")
                st.markdown(f"### {delta:+.2f}")
                st.caption("Comparaison directe des deux modeles")

# ---------------------------------------------------------------------------
# Tab 3: Cold start
# ---------------------------------------------------------------------------
with tab_cold:
    st.header("Recommander pour un nouvel utilisateur (cold start)")
    st.caption("Le modele hybride utilise les features (genre, age, occupation) sans aucun historique.")

    with st.container(border=True):
        st.subheader(":material/account_circle: Profil du nouvel utilisateur")
        c1, c2, c3 = st.columns(3)
        gender = c1.segmented_control("Genre", options=["F", "M"], default="F")
        age = c2.select_slider("Tranche d age",
                                options=[1, 18, 25, 35, 45, 50, 56], value=25)
        occ_label = c3.selectbox("Occupation",
                                 ["autre (0)", "academic (1)", "artist (2)", "clerical (3)",
                                  "student (4)", "customer service (5)", "doctor (6)",
                                  "executive (7)", "farmer (8)", "homemaker (9)",
                                  "lawyer (10)", "programmer (11)", "retired (12)",
                                  "sales (13)", "scientist (14)", "self-employed (15)",
                                  "technician (16)", "tradesman (17)", "unemployed (18)",
                                  "writer (19)"])
        occupation = int(occ_label.split("(")[1].rstrip(")"))
        top_k_c = st.slider("Top-K", 5, 20, 10)

    if st.button("Recommander (cold start)", type="primary", icon=":material/auto_awesome:"):
        with st.spinner("Chargement du modele hybride LightFM..."):
            lfm, _ = build_lightfm(meta, lfm_state)
        feats = [("gender", gender), ("age", str(age)), ("occupation", str(occupation))]
        recs = lfm.recommend_for_new_user(feats, top_k=top_k_c)
        if not recs:
            st.warning("Aucune recommandation.")
        else:
            st.subheader("Recommandations")
            cols = st.columns(2)
            for i, (iid, score) in enumerate(recs):
                row = movies[movies.movie_id == iid]
                if row.empty:
                    continue
                with cols[i % 2]:
                    render_movie_card(row.iloc[0], score=score)

# ---------------------------------------------------------------------------
# Tab 4: Compare
# ---------------------------------------------------------------------------
with tab_compare:
    st.header("Comparaison des modeles")
    st.caption("MovieLens 1M, temporal split 80/20, evaluation sur 1000 utilisateurs (sauf m4/m5 sur echantillon 100k).")

    cmp_df = pd.DataFrame({
        "Modele":  ["Popularity", "ItemItemCosine", "Surprise SVD", "FunkSVD (ours)",
                    "ALS-MF (ours)", "TwoTower", "NeuralCF", "LightFM hybride"],
        "RMSE":    [None, 0.955, 0.871, 0.902, 1.169, 1.063, 0.980, 0.972],
        "P@10":    [0.195, 0.223, 0.079, 0.064, 0.018, 0.009, 0.010, 0.002],
        "NDCG@10": [0.165, 0.207, 0.072, 0.055, 0.013, 0.014, 0.016, 0.002],
        "HitRate@10": [0.619, 0.645, 0.409, 0.338, 0.156, 0.077, 0.087, 0.020],
        "ColdStart": ["non", "non", "non", "non", "non", "non", "non", "oui"],
    })
    cmp_df = cmp_df.set_index("Modele")

    # Bar chart for ranking metrics
    st.subheader("Metriques de ranking (P@10, NDCG@10, HitRate@10)")
    rank_long = cmp_df.reset_index().melt(
        id_vars="Modele",
        value_vars=["P@10", "NDCG@10", "HitRate@10"],
        var_name="Metrique",
        value_name="Valeur",
    )
    st.bar_chart(rank_long, x="Modele", y="Valeur", color="Metrique", height=400)

    # Bar chart for RMSE
    st.subheader("RMSE (prediction de note, plus bas = meilleur)")
    rmse_df = cmp_df.dropna(subset=["RMSE"])[["RMSE"]].reset_index()
    st.bar_chart(rmse_df, x="Modele", y="RMSE", height=300, color="#d62728")

    # Table
    st.subheader("Tableau recapitulatif")
    numeric_cols = ["RMSE", "P@10", "NDCG@10", "HitRate@10"]
    styler = (cmp_df.style
              .highlight_max(axis=0, subset=["P@10", "NDCG@10", "HitRate@10"])
              .highlight_min(axis=0, subset=["RMSE"]))
    for col in numeric_cols:
        styler = styler.format({col: "{:.3f}"}, na_rep="-")
    st.dataframe(styler, width="stretch")

    # Conclusion in 3 cards
    st.subheader("Conclusions cles")
    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**:material/target: ItemItemCosine**")
            st.write("Imbattable sur le ranking (P@10 = 0.223) grace a son caractere non-parametrique et local. Simple mais tres efficace.")
    with c2:
        with st.container(border=True):
            st.markdown("**:material/calculate: Surprise SVD**")
            st.write("Meilleur RMSE / MAE (0.871). Implementation C optimisee, generalisation grace aux facteurs latents.")
    with c3:
        with st.container(border=True):
            st.markdown("**:material/person_add: LightFM hybride**")
            st.write("Seul modele a gerer le cold start. Combine interactions collaboratives + features demographiques et contenu.")

st.sidebar.markdown("---")
st.sidebar.caption("M1 cosine / M2 Surprise / M3 FunkSVD-ALS / M4 TwoTower-NeuralCF / M5 LightFM")
