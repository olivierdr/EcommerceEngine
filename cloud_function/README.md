# BigQuery Export Cloud Function

This Cloud Function exports prediction data from the front-end to BigQuery.

## Prerequisites

1. Google Cloud Project with BigQuery API enabled
2. BigQuery dataset `Ecommerce` created
3. BigQuery table `predictions` created with the correct schema
4. Service account with BigQuery Data Editor role

## Deployment

### 1. Set environment variables

```bash
export GCP_PROJECT="your-project-id"
export REGION="us-central1"  # or your preferred region
export FUNCTION_NAME="export-predictions-to-bigquery"
```

### 2. Deploy the function

```bash
gcloud functions deploy export-predictions-to-bigquery \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=export_to_bigquery \
  --trigger-http \
  --allow-unauthenticated \
  --set-env-vars BIGQUERY_DATASET=Ecommerce,BIGQUERY_TABLE=predictions \
  --memory=256MB \
  --timeout=60s
```

### 3. Get the function URL

After deployment, get the function URL:

```bash
gcloud functions describe export-predictions-to-bigquery \
  --gen2 \
  --region=$REGION \
  --format="value(serviceConfig.uri)"
```

## Testing

Test the function locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
functions-framework --target=export_to_bigquery --port=8080

# Test with curl
curl -X POST http://localhost:8080 \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "product_id": "test-123",
        "title": "Test Product",
        "description": "Test Description",
        "true_category_id": "cat1",
        "true_category_name": "Category 1",
        "true_category_path": "path/to/cat1",
        "predicted_category_id": "cat1",
        "predicted_category_name": "Category 1",
        "predicted_category_path": "path/to/cat1",
        "confidence": 0.95,
        "is_correct": true,
        "processing_time_ms": 100.5,
        "request_time_ms": 150.2,
        "timestamp": "2024-01-01T12:00:00Z",
        "api_url": "http://localhost:8000"
      }
    ]
  }'
```

## Security (Production)

For production, you should:

1. **Remove `--allow-unauthenticated`** and use authentication
2. **Add API key validation** in the function
3. **Use IAM** to restrict access
4. **Enable VPC connector** if needed

Example with authentication:

```bash
gcloud functions deploy export-predictions-to-bigquery \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=export_to_bigquery \
  --trigger-http \
  --no-allow-unauthenticated \
  --set-env-vars BIGQUERY_DATASET=Ecommerce,BIGQUERY_TABLE=predictions
```

## Environment Variables

- `GCP_PROJECT`: Your GCP project ID (auto-detected if running on GCP)
- `BIGQUERY_DATASET`: BigQuery dataset name (default: `Ecommerce`)
- `BIGQUERY_TABLE`: BigQuery table name (default: `predictions`)

