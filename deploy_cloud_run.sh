#!/bin/bash
# Script de déploiement sur Cloud Run

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"master-ai-cloud"}
SERVICE_NAME="ecommerce-classification-api"
REGION="europe-west1"
REPO_NAME="cloud-run-repo2"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "🚀 Déploiement sur Cloud Run..."
echo "   Project: ${PROJECT_ID}"
echo "   Service: ${SERVICE_NAME}"
echo "   Region: ${REGION}"
echo ""

# Vérifier que gcloud est configuré
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI n'est pas installé"
    exit 1
fi

# Vérifier que le modèle existe
if [ ! -f "results/classification/flat_model.pkl" ]; then
    echo "⚠️  Attention: Le modèle n'existe pas."
    echo "   Exécutez d'abord: python3 src/train.py"
    exit 1
fi

# Vérifier/créer le repository Artifact Registry
echo "🔍 Vérification du repository Artifact Registry..."
if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "📦 Création du repository Artifact Registry..."
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Repository pour Cloud Run FastAPI" \
        --project=${PROJECT_ID}
    echo "✓ Repository créé"
else
    echo "✓ Repository existe déjà"
fi

# Construire l'image Docker
echo "📦 Construction de l'image Docker..."
gcloud builds submit --tag ${IMAGE_NAME} --project ${PROJECT_ID}

# Déployer sur Cloud Run
echo "🚀 Déploiement sur Cloud Run..."
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
    --project ${PROJECT_ID}

# Obtenir l'URL du service
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format 'value(status.url)')

echo ""
echo "✅ Déploiement terminé!"
echo "   URL: ${SERVICE_URL}"
echo "   Health: ${SERVICE_URL}/health"
echo "   Docs: ${SERVICE_URL}/docs"
echo ""
echo "📊 Pour voir les métriques:"
echo "   https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics?project=${PROJECT_ID}"

