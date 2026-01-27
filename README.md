# Classification E-commerce

Système complet de classification automatique de produits e-commerce avec API REST et interface web interactive. Le projet combine un modèle de machine learning pour la classification, une API FastAPI pour l'exposition des services, et un frontend Next.js pour l'interaction utilisateur.

## Origine du projet

Ce projet a été développé pour répondre au besoin de classification automatique de produits e-commerce dans une taxonomie hiérarchique complexe. L'objectif principal est d'améliorer la visibilité des produits et d'assurer une expérience utilisateur cohérente en classant automatiquement les produits dans les bonnes catégories à partir de leurs informations textuelles (titre, description, marque, couleur).

Le projet s'articule autour de plusieurs besoins identifiés :

- **Classification automatique** : Prédire la catégorie feuille d'un produit parmi 100 catégories possibles à partir de ses informations textuelles
- **API REST** : Exposer le modèle de classification via une API pour permettre l'intégration dans d'autres systèmes
- **Interface web interactive** : Fournir une interface utilisateur pour tester le modèle, charger des datasets, visualiser les résultats et analyser les performances
- **Versioning des modèles et données** : Mettre en place un système de versioning pour suivre l'évolution des modèles entraînés et des datasets utilisés
- **Déploiement en production** : Déployer l'API sur Cloud Run et le frontend sur Firebase Hosting pour une disponibilité publique

## Architecture

### Vue d'ensemble

Le projet suit une architecture modulaire avec séparation claire entre le backend (API Python/FastAPI) et le frontend (Next.js/React), avec intégration des services Google Cloud Platform pour le stockage et le déploiement.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                       │
│  - Interface utilisateur interactive                        │
│  - Tests de produits                                        │
│  - Visualisation des résultats                              │
│  - Déployé sur Firebase Hosting                            │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────────┐
│                    API (FastAPI)                            │
│  - Endpoint /classify pour la classification                │
│  - Endpoint /testset pour charger le dataset                │
│  - Endpoint /category-names pour les métadonnées            │
│  - Métriques Prometheus                                     │
│  - Déployé sur Cloud Run                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│   Modèle ML  │ │  GCS      │ │  BigQuery  │
│  (Pickle)    │ │ (Storage) │ │ (Tracking) │
│              │ │           │ │            │
│ - flat_model │ │ - Modèles │ │ - Métriques│
│ - Embeddings │ │ - Datasets│ │ - Versions │
└──────────────┘ └───────────┘ └────────────┘
```

### Composants principaux

#### Backend (API FastAPI)

- **ClassificationModel** : Classe wrapper pour charger et utiliser le modèle de classification
  - Support du chargement depuis le système de fichiers local ou Google Cloud Storage (GCS)
  - Gestion automatique des embeddings avec sentence-transformers
  - Prédiction avec scores de confiance

- **Endpoints REST** :
  - `POST /classify` : Classification d'un produit (titre + description)
  - `GET /testset` : Récupération du dataset de test (CSV)
  - `GET /category-names` : Métadonnées des catégories avec exemples
  - `GET /health` : Vérification de l'état de santé
  - `GET /metrics` : Métriques Prometheus
  - `GET /docs` : Documentation interactive Swagger

- **Métriques Prometheus** :
  - Latence des requêtes (`api_request_duration_seconds`)
  - Throughput (`api_requests_total`)
  - Taux d'erreur (`api_errors_total`)
  - Score de confiance moyen (`api_confidence_score_average`)
  - Temps d'inférence (`api_inference_duration_seconds`)

#### Frontend (Next.js)

- **Pages principales** :
  - `/` : Page d'accueil avec testeur de produits et visualisation des résultats
  - `/categories` : Navigation dans toutes les catégories avec exemples

- **Composants** :
  - `ProductTester` : Interface pour tester des produits individuellement ou charger un dataset
  - `ResultsTable` : Tableau des résultats avec colonnes True Category, Predicted Category, Confidence, etc.
  - `StatsCards` : Cartes statistiques (accuracy, confiance moyenne, latence moyenne)
  - `Charts` : Graphiques de distribution de confiance, précision par catégorie, latence

- **Fonctionnalités** :
  - Chargement du testset depuis l'API
  - Test de produits aléatoires (10 par défaut)
  - Test manuel de produits individuels
  - Sauvegarde des résultats dans localStorage
  - Affichage des noms de catégories lisibles (résolution depuis l'API)

#### Stockage et versioning

- **Google Cloud Storage (GCS)** :
  - Stockage des modèles versionnés : `gs://bucket/models/{version}/model.pkl`
  - Stockage des datasets versionnés : `gs://bucket/datasets/{version}/{split}set.csv`
  - Stockage des métadonnées : `gs://bucket/models/{version}/category_names.json`

- **BigQuery** :
  - Tracking des versions de modèles
  - Métriques d'entraînement et d'évaluation
  - Historique des performances

- **Versioning** :
  - Format de version : `v1.0.0`, `v1.1.0`, etc.
  - Chaque version inclut le modèle, les métadonnées et les datasets associés
  - Traçabilité complète via BigQuery

### Environnements

Le projet supporte deux modes de fonctionnement :

- **Mode local** (`MODEL_SOURCE=local`) :
  - Charge le modèle depuis `results/classification/flat_model.pkl`
  - Charge les catégories depuis `results/audit/category_names.json`
  - Utilisé pour le développement et les tests locaux

- **Mode production** (`MODEL_SOURCE=gcs`) :
  - Charge le modèle depuis Google Cloud Storage
  - Version du modèle configurée via `MODEL_VERSION` (défaut: `v1.0.0`)
  - Utilisé pour les déploiements Cloud Run et la production

## Structure du projet

```
ClassificationEcommerce/
├── src/                          # Code source Python
│   ├── api.py                    # API FastAPI principale
│   ├── train.py                  # Script d'entraînement avec versioning
│   ├── evaluate.py               # Évaluation du modèle
│   ├── audit_taxonomy.py         # Audit de la taxonomie
│   ├── data/                     # Utilitaires de chargement de données
│   │   ├── loader.py             # Chargement local/GCS
│   │   └── testset.csv           # Dataset de test
│   ├── models/                   # Modèles de classification
│   │   └── flat_classifier.py    # Classifieur flat (baseline)
│   ├── training/                 # Utilitaires d'entraînement
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── tracking/                 # Tracking et versioning
│   │   ├── gcs.py                # Upload/download GCS
│   │   ├── bigquery.py           # Tracking BigQuery
│   │   └── vertex_ai.py         # Intégration Vertex AI
│   └── utils/                    # Utilitaires
│       ├── config.py             # Configuration GCP
│       └── category_names.py     # Gestion des noms de catégories
│
├── frontend-nextjs/              # Frontend Next.js
│   ├── app/                      # Pages App Router
│   │   ├── page.tsx              # Page d'accueil
│   │   ├── categories/           # Page catégories
│   │   ├── config.ts             # Configuration API URL
│   │   └── types.ts              # Types TypeScript
│   ├── components/               # Composants React
│   │   ├── ProductTester.tsx
│   │   ├── ResultsTable.tsx
│   │   ├── StatsCards.tsx
│   │   └── Charts.tsx
│   ├── scripts/                  # Scripts de test
│   │   ├── test-api-flow.mjs
│   │   └── test-parse-csv.mjs
│   ├── .env.production           # Configuration production
│   └── firebase.json             # Configuration Firebase Hosting
│
├── results/                      # Résultats générés (local)
│   ├── classification/
│   │   └── flat_model.pkl        # Modèle entraîné
│   └── audit/
│       └── category_names.json   # Noms de catégories
│
├── scripts/                      # Scripts de déploiement
│   ├── start_local.sh            # Démarrage local (API + Frontend)
│   ├── start_prod.sh             # Test production local
│   ├── deploy_cloud_run.sh       # Déploiement API Cloud Run
│   └── deploy_all.sh             # Déploiement complet (API + Frontend)
│
├── Dockerfile                    # Image Docker pour Cloud Run
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

## Technologies utilisées

### Backend

- **Python 3.11** : Langage principal
- **FastAPI** : Framework web pour l'API REST
- **scikit-learn** : Classification (Logistic Regression)
- **sentence-transformers** : Embeddings multilingues (`paraphrase-multilingual-MiniLM-L12-v2`)
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **prometheus-client** : Métriques Prometheus
- **google-cloud-storage** : Intégration GCS
- **google-cloud-bigquery** : Tracking BigQuery

### Frontend

- **Next.js 14** : Framework React avec App Router
- **TypeScript** : Typage statique
- **Tailwind CSS** : Styling
- **Recharts** : Visualisation de données
- **Firebase Hosting** : Déploiement frontend

### Infrastructure

- **Google Cloud Run** : Hébergement API (serverless)
- **Google Cloud Storage** : Stockage modèles et datasets
- **Google BigQuery** : Tracking et métriques
- **Firebase Hosting** : Hébergement frontend statique
- **Docker** : Containerisation pour Cloud Run

## Installation

### Prérequis

- Python 3.11+
- Node.js 18+
- Google Cloud SDK (`gcloud`)
- Firebase CLI (`firebase-tools`)
- Compte Google Cloud Platform configuré

### Installation des dépendances

**Backend :**

```bash
# Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch CPU-only (recommandé pour éviter les problèmes CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
pip install -r requirements.txt
```

**Frontend :**

```bash
cd frontend-nextjs
npm install
```

### Configuration Google Cloud

```bash
# Authentification
gcloud auth login

# Définir le projet par défaut
gcloud config set project master-ai-cloud

# Activer les APIs nécessaires
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### Configuration Firebase

```bash
cd frontend-nextjs
firebase login
firebase init hosting
```

## Utilisation

### Développement local

**Option 1 : Script automatique (recommandé)**

```bash
./start_local.sh
```

Ce script lance :
- L'API sur `http://localhost:8000` (mode développement avec reload)
- Le frontend sur `http://localhost:3000` (mode développement Next.js)
- Charge le modèle depuis le système de fichiers local

**Option 2 : Lancement manuel**

```bash
# Terminal 1 : API
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 : Frontend
cd frontend-nextjs
npm run dev
```

### Test production local

Pour tester le comportement en production (modèle depuis GCS) :

```bash
./start_prod.sh
```

Ce script :
- Lance l'API sur `http://localhost:8000` (sans reload, mode production)
- Build le frontend avec `.env.production`
- Sert les fichiers statiques sur `http://localhost:3001`
- Charge le modèle depuis GCS (`MODEL_SOURCE=gcs`)

### Entraînement du modèle

**Entraînement local :**

```bash
python3 src/train.py --version v1.0.0 --local-only
```

**Entraînement avec upload GCS/BigQuery :**

```bash
python3 src/train.py --version v1.0.0
```

Le script :
- Entraîne le modèle sur le dataset d'entraînement
- Évalue sur le dataset de test
- Sauvegarde le modèle localement dans `results/classification/`
- Upload vers GCS dans `gs://bucket/models/{version}/model.pkl`
- Enregistre les métriques dans BigQuery

### Audit de la taxonomie

```bash
python3 src/audit_taxonomy.py
```

Génère :
- Analyse de la structure hiérarchique
- Détection des incohérences
- Évaluation de la cohérence sémantique
- Génération des noms de catégories (`results/audit/category_names.json`)

### Déploiement

**Déploiement complet (API + Frontend) :**

```bash
./deploy_all.sh
```

Ce script :
1. Déploie l'API sur Cloud Run
2. Récupère l'URL de l'API déployée
3. Met à jour `frontend-nextjs/.env.production` avec l'URL API
4. Build et déploie le frontend sur Firebase Hosting

**Déploiement API uniquement :**

```bash
./deploy_cloud_run.sh
```

**Déploiement frontend uniquement :**

```bash
cd frontend-nextjs
npm run deploy:firebase
```

## Configuration

### Variables d'environnement

**API (Backend) :**

- `MODEL_SOURCE` : Source du modèle (`local` ou `gcs`, défaut: `local`)
- `MODEL_VERSION` : Version du modèle à charger depuis GCS (défaut: `v1.0.0`)
- `GOOGLE_CLOUD_PROJECT` : ID du projet GCP (défaut: `master-ai-cloud`)
- `GCS_BUCKET` : Nom du bucket GCS (défaut: `master-ai-cloud-ecommerce-ml`)

**Frontend :**

- `NEXT_PUBLIC_API_URL` : URL de l'API (défini dans `.env.production` pour le build)

### Fichiers de configuration

- `src/utils/config.py` : Configuration GCP (PROJECT_ID, REGION, BUCKET, etc.)
- `frontend-nextjs/.env.production` : Configuration production frontend
- `Dockerfile` : Configuration de l'image Docker pour Cloud Run
- `firebase.json` : Configuration Firebase Hosting

## Performance et métriques

### Métriques du modèle

- **Accuracy globale** : 77.47% sur le test set
- **Distribution de confiance** : 75.5% des produits avec confiance >= 0.5
- **Produits incertains** : 24.5% nécessitent validation humaine

### Métriques API (Prometheus)

Les métriques sont exposées sur `/metrics` :

- `api_request_duration_seconds` : Latence des requêtes
- `api_requests_total` : Nombre total de requêtes
- `api_errors_total` : Nombre d'erreurs
- `api_confidence_score_average` : Score de confiance moyen
- `api_inference_duration_seconds` : Temps d'inférence du modèle

### Visualisation

Les métriques peuvent être :
- Scrapées par Prometheus
- Visualisées dans Grafana
- Intégrées dans Google Cloud Monitoring

## Documentation complémentaire

- **API_README.md** : Documentation détaillée de l'API
- **SYNTHESIS.md** : Synthèse méthodologique et résultats détaillés
- **ARCHITECTURE.md** : Architecture détaillée du système
- **CLOUD_MONITORING.md** : Guide de monitoring Cloud

## Améliorations futures

### Techniques

- Migration vers des modèles plus sophistiqués (BERT fine-tuné, Transformers)
- Exploitation de la hiérarchie (approche top-down ou hybride)
- Intégration efficace de la marque et de la couleur
- Seuil de confiance adaptatif par catégorie

### Opérationnelles

- Détection automatique de nouvelles catégories
- Monitoring en temps réel des performances
- Workflow d'intégration des corrections humaines
- Active learning pour prioriser les produits à valider

## Licence

Ce projet est un projet de démonstration et d'apprentissage.
