# Authentication Setup for Cloud Function

## Overview

The Cloud Function now supports API Key authentication using Google Secret Manager.

## Setup Steps

### 1. Create and Store API Key

Run the setup script to generate and store an API key:

```bash
cd cloud_function
./setup_auth.sh
```

This will:
- Generate a random 64-character API key
- Store it in Google Secret Manager
- Grant the Cloud Function service account access to read it

**IMPORTANT**: Save the API key that is displayed - you'll need it for the front-end!

### 2. Deploy the Function

The deployment script will automatically detect if the API key secret exists:

```bash
./deploy_cloud_function.sh
```

- If secret exists: Function will be deployed with `--no-allow-unauthenticated` (requires API key)
- If secret doesn't exist: Function will be deployed with `--allow-unauthenticated` (development mode)

### 3. Use in Front-End

In the front-end, enter the API key in the "API Key" field when exporting to BigQuery.

The API key will be sent as a header: `X-API-Key: <your-api-key>`

## Manual Setup (Alternative)

If you prefer to set up manually:

### Create Secret

```bash
# Generate API key
API_KEY=$(openssl rand -hex 32)

# Create secret
echo -n "$API_KEY" | gcloud secrets create bigquery-export-api-key \
  --project=master-ai-cloud \
  --data-file=-

# Grant access to service account
gcloud secrets add-iam-policy-binding bigquery-export-api-key \
  --project=master-ai-cloud \
  --member="serviceAccount:189015650815-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Deploy with Authentication

```bash
gcloud functions deploy export-predictions-to-bigquery \
  --gen2 \
  --runtime=python311 \
  --region=europe-west1 \
  --source=. \
  --entry-point=export_to_bigquery \
  --trigger-http \
  --no-allow-unauthenticated \
  --set-env-vars BIGQUERY_DATASET=Ecommerce,BIGQUERY_TABLE=predictions,API_KEY_SECRET_NAME=bigquery-export-api-key \
  --memory=256MB \
  --timeout=60s
```

## Security Notes

1. **API Key Storage**: The API key is stored in Secret Manager, not in code
2. **Header vs Query**: API key can be sent as header (`X-API-Key`) or query parameter (`?api_key=...`)
3. **Development Mode**: If no API key is configured, the function allows unauthenticated access (for development only)
4. **Rotation**: To rotate the API key, create a new version in Secret Manager and update the front-end

## Testing

Test the authenticated endpoint:

```bash
# With API key in header
curl -X POST https://export-predictions-to-bigquery-pjplfwgp5q-ew.a.run.app \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"predictions": [...]}'

# With API key in query parameter
curl -X POST "https://export-predictions-to-bigquery-pjplfwgp5q-ew.a.run.app?api_key=YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"predictions": [...]}'
```


