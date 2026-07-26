# RecoSphere Web

Frontend Next.js du parcours films : inscription, onboarding, catalogue,
interactions, favoris et recommandations.

## Développement

```bash
npm install
npm run dev
```

Configurez `NEXT_PUBLIC_RECOSPHERE_API_URL` avec l’URL publique de l’API,
par défaut `http://localhost:8000`.

## Docker Compose

Depuis la racine du projet :

```bash
docker compose -f infra/docker-compose.yml up --build -d
```

Les ports Docker locaux sont configurables. Sur cette installation :

- frontend : `http://localhost:3100` ;
- API : `http://localhost:8010` ;
- Swagger : `http://localhost:8010/docs`.
