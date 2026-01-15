#!/bin/bash
# Cloud Run deployment script

set -e

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"master-ai-cloud"}
SERVICE_NAME="ecommerce-classification-api"
REGION="europe-west1"
REPO_NAME="cloud-run-repo2"
IMAGE_NAME="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${SERVICE_NAME}"

echo "Deploying to Cloud Run..."
echo "   Project: ${PROJECT_ID}"
echo "   Service: ${SERVICE_NAME}"
echo "   Region: ${REGION}"
echo ""

# Check gcloud is configured
if ! command -v gcloud &> /dev/null; then
    echo "ERROR: gcloud CLI is not installed"
    exit 1
fi

# Check model exists
if [ ! -f "results/classification/flat_model.pkl" ]; then
    echo "WARNING: Model does not exist."
    echo "   Run first: python3 src/train.py"
    exit 1
fi

# Check/create Artifact Registry repository
echo "Checking Artifact Registry repository..."
if ! gcloud artifacts repositories describe ${REPO_NAME} --location=${REGION} --project=${PROJECT_ID} &>/dev/null; then
    echo "Creating Artifact Registry repository..."
    gcloud artifacts repositories create ${REPO_NAME} \
        --repository-format=docker \
        --location=${REGION} \
        --description="Repository for Cloud Run FastAPI" \
        --project=${PROJECT_ID}
    echo "Repository created"
else
    echo "Repository already exists"
fi

# Build Docker image
echo "Building Docker image..."
gcloud builds submit --tag ${IMAGE_NAME} --project ${PROJECT_ID}

# Deploy to Cloud Run
echo "Deploying to Cloud Run..."
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

# Get service URL
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} --region ${REGION} --project ${PROJECT_ID} --format 'value(status.url)')

echo ""
echo "Deployment completed!"
echo "   URL: ${SERVICE_URL}"
echo "   Health: ${SERVICE_URL}/health"
echo "   Docs: ${SERVICE_URL}/docs"
echo ""
echo "To view metrics:"
echo "   https://console.cloud.google.com/run/detail/${REGION}/${SERVICE_NAME}/metrics?project=${PROJECT_ID}"
