#!/bin/bash
# Script to deploy the BigQuery export Cloud Function

set -e

# Configuration
FUNCTION_NAME="export-predictions-to-bigquery"
REGION="${REGION:-europe-west1}"
RUNTIME="python311"
MEMORY="256MB"
TIMEOUT="60s"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying Cloud Function: ${FUNCTION_NAME}${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    exit 1
fi

# Get current project
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
if [ -z "$PROJECT_ID" ]; then
    echo "Error: No GCP project configured. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Deploy the function
cd cloud_function

echo -e "${YELLOW}Deploying function...${NC}"
gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --runtime=$RUNTIME \
  --region=$REGION \
  --source=. \
  --entry-point=export_to_bigquery \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars BIGQUERY_DATASET=Ecommerce,BIGQUERY_TABLE=predictions \
  --memory=$MEMORY \
  --timeout=$TIMEOUT

# Get the function URL
echo -e "${YELLOW}Getting function URL...${NC}"
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME \
  --gen2 \
  --region=$REGION \
  --format="value(serviceConfig.uri)" 2>/dev/null)

if [ -n "$FUNCTION_URL" ]; then
    echo -e "${GREEN}✓ Function deployed successfully!${NC}"
    echo -e "${GREEN}Function URL: ${FUNCTION_URL}${NC}"
    echo ""
    echo "Add this URL to your front-end configuration."
else
    echo -e "${YELLOW}⚠ Could not retrieve function URL. Check the deployment status.${NC}"
fi

cd ..

