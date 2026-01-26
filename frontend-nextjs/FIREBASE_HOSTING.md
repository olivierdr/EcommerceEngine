# Déploiement sur Firebase Hosting

Le front est configuré pour un **export statique** (`output: 'export'`) et pour être servi par Firebase Hosting.

## Prérequis

1. **Node.js** (déjà utilisé pour le front)
2. **Firebase CLI** :
   ```bash
   npm install -g firebase-tools
   firebase login
   ```
3. **Projet Firebase** : créez-en un sur [Firebase Console](https://console.firebase.google.com/) ou réutilisez un projet GCP. Puis associez-le :
   ```bash
   firebase use <votre-project-id>
   ```
   (ou éditez `.firebaserc` et remplacez `ecommerce-classification` par votre project ID)

## Vérifier que le build fonctionne (en amont)

Depuis `frontend-nextjs/` :

```bash
npm ci
npm run build
```

Un dossier `out/` doit être généré. Vous pouvez servir en local pour tester :

```bash
npx serve out
```

Ouvrez http://localhost:3000 (ou le port indiqué), vérifiez que l’app s’affiche et que le champ « API URL » fonctionne (pointez vers votre API Cloud Run en prod).

## Déployer sur Firebase Hosting

Toujours depuis `frontend-nextjs/` :

```bash
npm run build
firebase deploy
```

L’URL publique sera du type :  
`https://<project-id>.web.app` ou `https://<project-id>.firebaseapp.com`.

## Après déploiement

Sur le site en ligne, renseignez l’URL de votre API Cloud Run dans le champ « API URL » (ou prévoyez une variable d’environnement au build si vous voulez une valeur par défaut).
