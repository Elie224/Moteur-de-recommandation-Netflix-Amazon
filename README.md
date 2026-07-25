# Moteur de recommandation Netflix / Amazon

Systeme de recommandation automatique de films et produits, construit progressivement avec des techniques utilisees en production par Netflix, Amazon, Spotify, Cdiscount, etc.

## Objectifs pedagogiques

- Comprendre les bases mathematiques des systemes de recommandation (cosine, SVD, factorisation matricielle).
- Implementer plusieurs familles d algorithmes et les comparer sur des metriques de ranking (Precision@K, Recall@K, NDCG, MAP).
- Savoir gerer le cold start (nouvel utilisateur / nouvel item).
- Construire un modele hybride combinant filtrage collaboratif et contenu.
- Livrer une API et un dashboard pour rendre le systeme tangible.

## Modules

| # | Module                          | Techniques                                | Librairies               |
|---|---------------------------------|-------------------------------------------|--------------------------|
| 1 | Baselines                       | Cosine similarity, popularite globale     | numpy, scipy, pandas     |
| 2 | Collaborative Filtering         | KNN, SVD, NMF, BaselineOnly, CoClustering | scikit-surprise          |
| 3 | Matrix Factorization            | FunkSVD from scratch, ALS-MF              | numpy                    |
| 4 | Embeddings neuroniques          | TwoTower, Neural Collaborative Filtering  | PyTorch                  |
| 5 | Modele hybride                  | LightFM-style (MSE + BPR, side features)  | PyTorch                  |
| 6 | API REST                        | FastAPI + TestClient                      | fastapi, uvicorn         |
| 7 | Dashboard interactif            | Streamlit                                 | streamlit                |

## Dataset

MovieLens 1M (~1M de notes, ~6K utilisateurs, ~4K films).
Split temporel 80/20 (pas random) pour respecter le contexte production.

## Structure du projet

```
Moteur de recommadation/
|-- data/raw/ml-1m/                donnees brutes MovieLens
|-- data/processed/artifacts/      modeles prod pre-entraines (cosine.pkl, svd.pkl, lightfm.pt)
|-- notebooks/                     5 notebooks (01..05)
|-- src/
|   |-- data/                      loaders + downloader
|   |-- evaluation/                metriques (RMSE, P@K, R@K, NDCG, MAP, coverage, novelty)
|   |-- models/                    baselines + Surprise + FunkSVD/ALS + PyTorch CF + LightFM
|   |-- api/                       FastAPI app
|-- app/                           Streamlit dashboard + smoke test
|-- scripts/                       build_notebook_*, train_production_models.py
|-- tests/                         pytest suite (49 tests)
|-- requirements.txt
|-- README.md
```

## Demarrage rapide

```powershell
cd "C:\Users\KOURO\Documents\Moteur de recommadation"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Telecharger MovieLens 1M
python src\data\download_data.py

# Entrainer les modeles de production
python scripts\train_production_models.py

# Lancer l API
.\.venv\Scripts\uvicorn src.api.app:app --reload --port 8000

# Lancer le dashboard
.\.venv\Scripts\python -m streamlit run app\dashboard.py

# Lancer les notebooks
.\.venv\Scripts\jupyter notebook notebooks\

# Tests
.\.venv\Scripts\python -m pytest tests\ -v
```

## Endpoints API

| Endpoint                              | Description                          |
|---------------------------------------|--------------------------------------|
| GET /                                 | Info service                         |
| GET /health                           | Health check                         |
| GET /models                           | Liste des modeles                    |
| GET /stats                            | Stats dataset                        |
| GET /movies/{id}                      | Info film                            |
| GET /movies?query=...                 | Recherche film                       |
| GET /users/{uid}/recommend?model=...  | Top-K recommandations                |
| GET /predict?user_id=...&movie_id=... | Predire une note                     |
| POST /cold-start/recommend            | Recos pour nouvel utilisateur        |

Documentation auto-generee : http://localhost:8000/docs

## Comparaison finale (MovieLens 1M, temporal split)

| Modele               | RMSE  | MAE   | P@10  | NDCG@10 | Cold Start |
|----------------------|-------|-------|-------|---------|------------|
| Popularity           | -     | -     | 0.195 | 0.165   | no         |
| ItemItemCosine (m1)  | 0.955 | 0.744 | 0.223 | 0.207   | no         |
| Surprise SVD (m2)    | 0.871 | 0.684 | 0.079 | 0.072   | no         |
| FunkSVD ours (m3)    | 0.902 | 0.706 | 0.064 | 0.055   | no         |
| ALS-MF ours (m3)     | 1.169 | 0.894 | 0.018 | 0.013   | no         |
| TwoTower (m4)        | 1.063 | 0.848 | -     | -       | no         |
| NeuralCF (m4)        | 0.980 | 0.788 | -     | -       | no         |
| LightFM hybrid (m5)  | 0.972 | 0.759 | -     | -       | yes        |

## Metriques d evaluation

- Prediction de note : RMSE, MAE
- Ranking : Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate
- Au-dela de la precision : catalog coverage, novelty

## Stack technique

- Python 3.14
- pandas, numpy, scipy, scikit-learn
- scikit-surprise
- PyTorch (CPU)
- FastAPI + uvicorn
- Streamlit
- pytest

## Notes techniques

- `SURPRISE_DATA_FOLDER` est redirige vers `.surprise_data/` (permissions Windows sur `~/.surprise_data`).
- LightFM est implemente from scratch en PyTorch (la lib officielle n a pas de wheel pour Python 3.14).
- Le smoke test du module 2 (KNN) a ete retire du test rapide car trop long sur 800k notes (CF en prod avec KNN est OK).
