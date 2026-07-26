# RecoSphere

Plateforme de recommandation multi-domaines pour les films et les produits.
RecoSphere sépare le catalogue unifié, les intégrations externes, l’API, le
frontend et le moteur de recommandation afin de pouvoir ajouter de nouvelles
sources sans modifier le cœur du système.

## Parcours actuel

```text
TMDB
  → validation et normalisation
  → PostgreSQL
  → API FastAPI
  → frontend Next.js
  → interactions utilisateur
  → recommandations personnalisées
```

Le parcours films disponible est :

```text
inscription → connexion → onboarding → catalogue → like/dislike/favori
→ recommandations
```

L’intégration eBay est présente dans le code mais reste volontairement en
attente. Anthropic est réservé aux explications et à l’enrichissement des
préférences ; le classement ne dépend pas d’un LLM.

## Architecture

```text
RecoSphere/
├── apps/
│   ├── api/                         API FastAPI active
│   │   ├── alembic/                 migrations PostgreSQL
│   │   ├── src/recommender_api/
│   │   │   ├── integrations/tmdb/   client, schémas, mapper
│   │   │   ├── integrations/ebay/   intégration en attente
│   │   │   ├── services/             catalogue, interactions, recommandations
│   │   │   └── workers/              synchronisations CLI
│   │   └── tests/
│   └── web/                         frontend Next.js actif
│       ├── src/app/                 pages films, auth, favoris, admin
│       ├── src/components/
│       └── src/lib/
├── packages/
│   └── recommendation_engine/       moteur Python réutilisable
├── infra/
│   └── docker-compose.yml            PostgreSQL, API et frontend
├── notebooks/                        laboratoire MovieLens historique
├── src/                              anciens modèles et évaluations
├── app/                              ancien dashboard Streamlit
├── scripts/                          anciens scripts d’entraînement
└── data/                             données et artefacts locaux ignorés
```

`apps/api`, `apps/web` et `packages/recommendation_engine` constituent la
plateforme active. Les notebooks, `src/` et `app/` documentent les expériences
MovieLens et restent séparés du parcours applicatif principal.

## Démarrage avec Docker

Prérequis : Docker Desktop démarré.

Depuis la racine du dépôt :

```powershell
docker compose -f infra/docker-compose.yml up --build -d
```

Sur la configuration locale actuelle :

| Service | Adresse |
|---|---|
| Frontend | http://localhost:3100 |
| API | http://localhost:8010 |
| Swagger | http://localhost:8010/docs |
| PostgreSQL | localhost:5432 |

Les ports hôtes peuvent être surchargés avec `RECOSPHERE_API_HOST_PORT` et
`RECOSPHERE_WEB_HOST_PORT`. Le frontend utilise `localhost:8010` dans le
navigateur et `api:8000` pour ses appels internes au réseau Docker.

Commandes utiles :

```powershell
docker compose -f infra/docker-compose.yml ps
docker compose -f infra/docker-compose.yml logs -f api
docker compose -f infra/docker-compose.yml logs -f web
docker compose -f infra/docker-compose.yml down
```

## Configuration

Copiez les variables nécessaires dans le fichier `.env` local. Ne commitez
jamais ce fichier : il peut contenir des tokens et des secrets.

```env
RECOSPHERE_TMDB_ACCESS_TOKEN=<API Read Access Token TMDB>
RECOSPHERE_TMDB_BASE_URL=https://api.themoviedb.org/3
RECOSPHERE_TMDB_IMAGE_BASE_URL=https://image.tmdb.org/t/p
RECOSPHERE_TMDB_IMAGE_SIZE=w500
RECOSPHERE_TMDB_LANGUAGE=fr-FR
RECOSPHERE_TMDB_REGION=FR
RECOSPHERE_TMDB_SYNC_MAX_PAGES=5
RECOSPHERE_TMDB_REQUEST_TIMEOUT=20
```

Le service API peut démarrer sans token TMDB, mais la synchronisation retournera
une erreur de configuration tant qu’un vrai **API Read Access Token** n’est pas
configuré.

## Synchroniser TMDB

La synchronisation crée un `SyncRun`, déduplique les identifiants TMDB et met à
jour `CatalogItem` ainsi que `Movie` sans créer de doublons.

Depuis Docker, pour un premier lot limité :

```powershell
docker compose -f infra/docker-compose.yml exec -T api `
  python -m recommender_api.workers.sync_tmdb --max-pages 1
```

L’endpoint administrateur équivalent est :

```http
POST /api/v1/admin/sync/tmdb
```

Après une synchronisation, les films sont disponibles avec :

```http
GET /api/v1/catalog?item_type=movie
GET /api/v1/catalog/{item_id}
```

## API principale

Toutes les routes applicatives sont préfixées par `/api/v1`.

| Route | Usage |
|---|---|
| `POST /auth/register` | créer un compte |
| `POST /auth/login` | se connecter |
| `GET /auth/me` | utilisateur courant |
| `GET /catalog` | catalogue films/produits |
| `GET /catalog/{id}` | détail d’un item actif |
| `POST /interactions` | impression, vue, like, dislike, etc. |
| `GET /favorites` | favoris de l’utilisateur |
| `POST /favorites/{id}` | ajouter/retirer un favori |
| `POST /onboarding` | préférences initiales |
| `GET /recommendations/{item_type}` | recommandations hybrides |
| `GET /admin/metrics` | métriques administrateur |
| `POST /admin/sync/tmdb` | synchronisation TMDB administrateur |

## Moteur de recommandation

Le package `recommendation_engine` est indépendant de FastAPI et de SQLAlchemy.
Le moteur hybride V1 combine actuellement :

```text
item cosine       35 %
content based     30 %
recent activity   20 %
popularity        15 %
```

Les frames SQL filtrent le domaine demandé et les items actifs. Les favoris
alimentent le contexte utilisateur ; les signaux forts (`like`, `dislike`,
`rating`, etc.) sont traités comme des préférences actives, tandis qu’une
simple vue ou impression ne retire pas définitivement l’item du catalogue de
recommandation.

Le laboratoire historique contient aussi des implémentations et comparaisons
MovieLens : popularité, cosine item-item, SVD, FunkSVD, ALS, Two-Tower,
NeuralCF et modèle hybride de type LightFM.

## Tests et validation

Tests API et intégrations sans appel TMDB/eBay réel :

```powershell
python -m pytest apps/api/tests -q
python -m pytest tests -q
```

Vérification frontend :

```powershell
cd apps/web
npm ci
npm run typecheck
npm run build
```

PostgreSQL est utilisé par Docker en environnement applicatif. Les tests
rapides utilisent SQLite ; les migrations de production sont gérées par
Alembic dans `apps/api/alembic`.

## Sécurité et limites actuelles

- Les secrets restent dans `.env` et ne doivent jamais être poussés.
- Le JWT stocké côté frontend est acceptable pour le MVP ; un cookie HTTP-only
  est recommandé avant un déploiement public.
- Les valeurs par défaut de Compose sont destinées au développement local.
- Le calcul du moteur est encore effectué à la demande ; un service d’artefacts
  versionnés et un cache sont les prochaines étapes de scalabilité.
- eBay reste en pause jusqu’à validation complète du parcours films.

## Prochaines étapes

1. Ajouter une CI GitHub Actions : tests backend, typecheck/build frontend,
   migrations et validation Compose.
2. Versionner les modèles entraînés et éviter le réentraînement à chaque
   requête de recommandation.
3. Normaliser les scores hybrides et mesurer Precision@K, Recall@K, NDCG,
   couverture, diversité et nouveauté.
4. Ajouter les tests PostgreSQL et Playwright du parcours utilisateur.
5. Reprendre eBay après stabilisation et mesure du parcours films.
