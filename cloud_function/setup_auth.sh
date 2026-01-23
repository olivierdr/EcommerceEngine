#!/bin/bash
# Script to set up API key authentication for Cloud Function

set -e

PROJECT_ID="${GCP_PROJECT:-master-ai-cloud}"
SECRET_NAME="bigquery-export-api-key"
REGION="${REGION:-europe-west1}"

echo "Setting up API key authentication for Cloud Function..."

# Generate a random API key
API_KEY=$(openssl rand -hex 32)
echo "Generated API key: ${API_KEY:0:8}...${API_KEY: -8}"

# Check if secret already exists
if gcloud secrets describe $SECRET_NAME --project=$PROJECT_ID &>/dev/null; then
    echo "Secret $SECRET_NAME already exists. Updating..."
    echo -n "$API_KEY" | gcloud secrets versions add $SECRET_NAME \
        --project=$PROJECT_ID \
        --data-file=-
else
    echo "Creating secret $SECRET_NAME..."
    echo -n "$API_KEY" | gcloud secrets create $SECRET_NAME \
        --project=$PROJECT_ID \
        --data-file=-
fi

# Grant Cloud Function service account access to the secret
SERVICE_ACCOUNT="189015650815-compute@developer.gserviceaccount.com"
echo "Granting service account access to secret..."
gcloud secrets add-iam-policy-binding $SECRET_NAME \
    --project=$PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor"

echo ""
echo "✓ API key configured successfully!"
echo ""
echo "IMPORTANT: Save this API key securely:"
echo "API_KEY=$API_KEY"
echo ""
echo "Add it to your Cloud Function environment variable or use it in the front-end."
echo "To use it in the front-end, add it as a header: X-API-Key: $API_KEY"

