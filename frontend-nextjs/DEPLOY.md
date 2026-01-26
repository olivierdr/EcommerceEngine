# Local vs prod : pointer le front vers la bonne API

L’URL de l’API par défaut est fixée par la variable `NEXT_PUBLIC_API_URL` (ou localhost si elle est absente).

## Local (API en local)

- **Commande** : `npm run dev`
- **Comportement** : aucune variable nécessaire. L’URL par défaut est `http://localhost:8000`.
- **À faire** : lancer l’API en local (ex. `uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`), puis ouvrir le front.

## Prod (front sur Firebase, API sur Cloud Run)

- **Commande** : soit `npm run deploy:firebase`, soit `npm run deploy:prod`.
- **Comportement** : au **build**, Next lit `NEXT_PUBLIC_API_URL`. Si elle est définie, le front utilisera cette URL par défaut une fois déployé sur Firebase.
- **À faire** :
  1. **Option A** : copier `.env.production.example` en `.env.production`, y mettre l’URL Cloud Run (sans slash final), puis lancer `npm run deploy:firebase` (ou `npm run deploy:prod`).
  2. **Option B** : ne pas créer `.env.production` et passer la variable à la main :  
     `NEXT_PUBLIC_API_URL=https://ton-api-xxx.run.app npm run deploy:firebase`

## Récap des commandes

| Contexte | Commande | API URL par défaut |
|----------|----------|--------------------|
| Local    | `npm run dev` | `http://localhost:8000` |
| Prod     | `NEXT_PUBLIC_API_URL=https://xxx.run.app npm run deploy:firebase` ou `.env.production` + `npm run deploy:firebase` | Celle définie dans la variable / le fichier |

Les scripts `build:prod` et `deploy:prod` sont des alias pour `build` et `deploy:firebase` ; ils servent à rappeler qu’en prod il faut définir `NEXT_PUBLIC_API_URL` avant de builder.
