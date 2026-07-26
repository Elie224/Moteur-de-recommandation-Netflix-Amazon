# RecoSphere API

API FastAPI du catalogue unifié et du moteur de recommandation RecoSphere.

Les variables d'environnement utilisent le préfixe `RECOSPHERE_`, par
exemple `RECOSPHERE_DATABASE_URL` et `RECOSPHERE_JWT_SECRET`.

La base par défaut est PostgreSQL :
`postgresql+psycopg://recosphere:recosphere@localhost:5432/recosphere`.
En production, définissez `RECOSPHERE_DATABASE_URL` avec les identifiants
réels du serveur PostgreSQL.

Pour activer la synchronisation eBay, renseignez `RECOSPHERE_EBAY_CLIENT_ID`
et `RECOSPHERE_EBAY_CLIENT_SECRET`. Le secret reste côté API et ne doit
jamais être utilisé dans une variable `NEXT_PUBLIC_*`.

Le fournisseur IA est Anthropic. Configurez `RECOSPHERE_ANTHROPIC_API_KEY` et,
si nécessaire, `RECOSPHERE_ANTHROPIC_MODEL` (Claude). Aucune clé OpenAI n’est
requise.
