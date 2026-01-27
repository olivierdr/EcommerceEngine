#!/bin/bash
# Déploie l'API (Cloud Run) puis le frontend (Firebase Hosting) en une seule commande.
# Met automatiquement à jour .env.production avec l'URL de l'API déployée.

set -e

cd "$(dirname "$0")"
ROOT="$PWD"

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"master-ai-cloud"}
SERVICE_NAME="ecommerce-classification-api"
REGION="europe-west1"
REPO_NAME="cloud-run-repo2"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "🚀 Déploiement complet (API + Frontend)"
echo "========================================"
echo ""

# Vérifications préalables
if ! command -v gcloud &> /dev/null; then
    echo "❌ ERROR: gcloud CLI is not installed"
    exit 1
fi

if ! command -v firebase &> /dev/null; then
    echo "❌ ERROR: Firebase CLI is not installed"
    echo "   Install: npm install -g firebase-tools"
    exit 1
fi

# Vérification authentification Google Cloud
echo "Vérification authentification Google Cloud..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null | grep -q .; then
    echo "❌ ERROR: Aucun compte Google Cloud authentifié"
    echo "   Exécutez: gcloud auth login"
    exit 1
fi

# Vérifier que le projet est configuré
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ -z "${CURRENT_PROJECT}" ]; then
    echo "❌ ERROR: Aucun projet GCP configuré"
    echo "   Exécutez: gcloud config set project ${PROJECT_ID}"
    exit 1
fi

echo "✓ Google Cloud authentifié (projet: ${CURRENT_PROJECT})"

# Vérification authentification Firebase
echo "Vérification authentification Firebase..."
if ! firebase projects:list &>/dev/null; then
    echo "❌ ERROR: Authentification Firebase échouée ou expirée"
    echo "   Exécutez: firebase login --reauth"
    exit 1
fi

echo "✓ Firebase authentifié"
echo ""

# ==================== ÉTAPE 1: Déploiement API ====================
echo "📦 ÉTAPE 1/2: Déploiement API sur Cloud Run..."
echo ""

# Check model exists (avertissement seulement, car en prod on utilise GCS)
if [ ! -f "results/classification/flat_model.pkl" ]; then
    echo "⚠️  WARNING: Modèle local absent (normal si MODEL_SOURCE=gcs)"
fi

# Check testset exists
if [ ! -f "src/data/testset.csv" ]; then
    echo "⚠️  WARNING: src/data/testset.csv not found. /testset retournera 404."
fi

# Check/create Artifact Registry repository
echo "Vérification Artifact Registry..."
if gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "✓ Repository existe déjà"
else
    echo "Création du repository..."
    # Désactiver temporairement set -e pour cette commande
    set +e
    create_out=$(gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Repository for Cloud Run FastAPI" \
        --project=${PROJECT_ID} 2>&1)
    create_rc=$?
    set -e
    if [ ${create_rc} -eq 0 ]; then
        echo "✓ Repository créé"
    elif echo "${create_out}" | grep -q "ALREADY_EXISTS"; then
        echo "✓ Repository existe déjà (continuing)"
    else
        echo "❌ Erreur création repository:"
        echo "${create_out}"
        exit 1
    fi
fi

# Build Docker image
echo ""
echo "Build de l'image Docker..."
gcloud builds submit --tag ${IMAGE_NAME} --project ${PROJECT_ID}

# Deploy to Cloud Run
echo ""
echo "Déploiement sur Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --timeout 300 \
    --min-instances 0 \
    --max-instances 10 \
    --set-env-vars MODEL_SOURCE=gcs,MODEL_VERSION=v1.0.0 \
    --project ${PROJECT_ID}

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format 'value(status.url)')

if [ -z "${SERVICE_URL}" ]; then
    echo "❌ ERROR: Impossible de récupérer l'URL du service Cloud Run"
    exit 1
fi

echo ""
echo "✓ API déployée avec succès!"
echo "   URL: ${SERVICE_URL}"
echo ""

# ==================== ÉTAPE 2: Mise à jour .env.production ====================
echo "📝 ÉTAPE 1.5/2: Mise à jour .env.production avec l'URL de l'API..."
ENV_FILE="frontend-nextjs/.env.production"

# Sauvegarder l'ancien fichier si il existe
if [ -f "${ENV_FILE}" ]; then
    cp "${ENV_FILE}" "${ENV_FILE}.backup"
    echo "✓ Backup créé: ${ENV_FILE}.backup"
fi

# Créer/mettre à jour .env.production
cat > "${ENV_FILE}" << EOF
# URL API Cloud Run (mise à jour automatiquement par deploy_all.sh)
# Utilisé au build pour Firebase (npm run build puis firebase deploy).
# En local (npm run dev), ce fichier n'est pas lu → défaut = http://localhost:8000
NEXT_PUBLIC_API_URL=${SERVICE_URL}
EOF

echo "✓ ${ENV_FILE} mis à jour avec: ${SERVICE_URL}"
echo ""

# ==================== ÉTAPE 3: Déploiement Frontend ====================
echo "🌐 ÉTAPE 2/2: Déploiement Frontend sur Firebase Hosting..."
echo ""

cd frontend-nextjs

# Vérifier que firebase.json existe
if [ ! -f "firebase.json" ]; then
    echo "❌ ERROR: firebase.json not found in frontend-nextjs/"
    echo "   Configure Firebase Hosting first: firebase init hosting"
    exit 1
fi

# Déployer sur Firebase
echo "Build et déploiement..."
npm run deploy:firebase

cd ..

# ==================== RÉSUMÉ ====================
echo ""
echo "========================================"
echo "✅ Déploiement complet terminé!"
echo ""
echo "📡 API (Cloud Run):"
echo "   ${SERVICE_URL}"
echo "   Health: ${SERVICE_URL}/health"
echo "   Docs: ${SERVICE_URL}/docs"
echo ""
echo "🌐 Frontend (Firebase Hosting):"
echo "   Vérifiez l'URL dans la sortie Firebase ci-dessus"
echo "   Ou: firebase hosting:sites:list"
echo ""
echo "📊 Métriques API:"
echo "   https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics?project=${PROJECT_ID}"
echo ""
