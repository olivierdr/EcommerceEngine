# Cloud Monitoring and Observability Guide

This guide explains how to use Cloud Monitoring and Cloud Trace to observe the classification API.

## Prerequisites

1. **GCP Project configured** with the following APIs enabled:
   - Cloud Run API
   - Cloud Monitoring API
   - Cloud Trace API
   - Cloud Build API

2. **Authentication**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Permissions**: The Cloud Run service must have permissions:
   - `roles/cloudtrace.agent` (for Cloud Trace)
   - `roles/monitoring.metricWriter` (for Cloud Monitoring)

## Deployment

1. **Deploy the API on Cloud Run**:
   ```bash
   chmod +x deploy_cloud_run.sh
   export GOOGLE_CLOUD_PROJECT=your-project-id
   ./deploy_cloud_run.sh
   ```

2. **Verify instrumentation is working**:
   - The API automatically detects running on GCP via the `GOOGLE_CLOUD_PROJECT` variable
   - Metrics are exported every 60 seconds
   - Traces are sent in batches

## Cloud Monitoring Dashboard

### Create a Custom Dashboard

1. **Access Cloud Monitoring**:
   - GCP Console → Monitoring → Dashboards
   - Click "Create Dashboard"

2. **Available Metrics**:

   #### Latency (Request Duration)
   - **Metric**: `custom.googleapis.com/opentelemetry/api_request_duration_seconds`
   - **Recommended chart**: Line chart with percentiles (p50, p95, p99)
   - **Filters**: `endpoint="/classify"`

   #### Throughput (Request Rate)
   - **Metric**: `custom.googleapis.com/opentelemetry/api_requests_total`
   - **Recommended chart**: Line chart (rate per second)
   - **Filters**: `endpoint="/classify"`, `status="2xx"`

   #### Error Rate
   - **Metric**: `custom.googleapis.com/opentelemetry/api_errors_total`
   - **Recommended chart**: Line chart
   - **Filters**: `error_type="5xx"` or `error_type="4xx"`

   #### Confidence Score
   - **Metric**: `custom.googleapis.com/opentelemetry/api_confidence_score_average`
   - **Recommended chart**: Line chart
   - **Alert threshold**: < 0.7

   #### Inference Time
   - **Metric**: `custom.googleapis.com/opentelemetry/api_inference_duration_seconds`
   - **Recommended chart**: Line chart with percentiles

### Recommended Dashboard (5 widgets)

1. **Latency p95** (red line at 500ms)
2. **Throughput** (requests/second)
3. **Error rate** (red line at 1%)
4. **Average confidence score**
5. **Inference time p95**

## Cloud Trace

### View Traces

1. **Access Cloud Trace**:
   - GCP Console → Trace → Trace List

2. **Filter traces**:
   - Service: `ecommerce-classification-api`
   - Endpoint: `/classify`

3. **Available Spans**:
   - `POST /classify`: Main request span
   - `model.predict`: Span for complete prediction
   - `model.embedding`: Span for embedding generation
   - `model.classify`: Span for classification

4. **Analyze Performance**:
   - Identify slow requests (> 1s)
   - See which step takes the most time (embedding vs classification)
   - Analyze errors with stack traces

## Alerts

### Create Alerts

1. **Access Alerts**:
   - GCP Console → Monitoring → Alerting → Create Policy

2. **Recommended Alerts**:

   #### High Latency
   - **Condition**: `api_request_duration_seconds` p95 > 500ms
   - **Duration**: 5 minutes
   - **Notification**: Email or Slack

   #### High Error Rate
   - **Condition**: `api_errors_total` rate > 1% of requests
   - **Duration**: 5 minutes

   #### Low Confidence Score
   - **Condition**: `api_confidence_score_average` < 0.7
   - **Duration**: 10 minutes

   #### Model Unavailable
   - **Condition**: Health check `/health` returns `unhealthy`
   - **Duration**: 1 minute

## Prometheus Metrics (local)

For local development, Prometheus metrics remain available at `/metrics`:
- Compatible with Grafana
- Standard Prometheus format

## Troubleshooting

### Metrics Not Appearing

1. Verify `GOOGLE_CLOUD_PROJECT` is defined in Cloud Run
2. Check service IAM permissions
3. Check Cloud Run logs for export errors

### Traces Not Appearing

1. Verify Cloud Trace API is enabled
2. Check `roles/cloudtrace.agent` permissions
3. Wait a few minutes (traces sent in batches)

### Performance

- Metrics are exported every 60 seconds (configurable)
- Traces are sent in batches (latency ~10-30 seconds)
- Minimal impact on API performance
