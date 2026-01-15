# FastAPI - E-commerce Classification

REST API for e-commerce product classification service with basic instrumentation.

## Quick Start

### 1. Install Dependencies

**Option A: Automatic script (recommended)**
```bash
./install.sh
```

**Option B: Manual installation**
```bash
# Create/activate venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch CPU-only first (avoids CUDA issues)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

> **Note**: PyTorch is installed in CPU-only mode to avoid errors with `nvidia_cublas_cu12`. If you need GPU support, install PyTorch with CUDA separately.

### 2. Train the Model (if not already done)

```bash
python3 src/train.py
```

### 3. Start the API

```bash
./start_api.sh
```

Or manually:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

The API will be accessible at: http://localhost:8000

## Endpoints

### `POST /classify`
Classify a product into a category.

**Request:**
```json
{
  "title": "Samsung Galaxy S21",
  "description": "Android smartphone with 6.2 inch screen"
}
```

**Response:**
```json
{
  "category_id": "12345",
  "category_path": "Electronics > Smartphones > Samsung",
  "confidence": 0.87,
  "processing_time_ms": 245.3
}
```

### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /metrics`
Prometheus endpoint for metrics.

### `GET /docs`
Interactive Swagger UI documentation.

## Metrics (5 Key Metrics)

The API exposes 5 main metrics via Prometheus:

1. **`api_request_duration_seconds`** (Histogram)
   - Request latency per endpoint and status

2. **`api_requests_total`** (Counter)
   - Throughput: total number of requests

3. **`api_errors_total`** (Counter)
   - Error rate: 4xx, 5xx errors, exceptions

4. **`api_confidence_score_average`** (Gauge)
   - Average confidence score of predictions

5. **`api_inference_duration_seconds`** (Histogram)
   - Model inference time

## Usage Examples

### With curl

```bash
# Classify a product
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "iPhone 14 Pro",
    "description": "Apple smartphone with A16 chip"
  }'

# Check health
curl http://localhost:8000/health

# Get metrics
curl http://localhost:8000/metrics
```

### With Python

```python
import requests

# Classify a product
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "title": "MacBook Pro 16",
        "description": "Apple laptop with M2 chip"
    }
)
print(response.json())
```

## Metrics Visualization

Metrics are in Prometheus format and can be:
- Scraped by Prometheus
- Visualized in Grafana
- Integrated with Cloud Monitoring (GCP)

### PromQL Query Examples

```promql
# Latency P95
histogram_quantile(0.95, api_request_duration_seconds_bucket)

# Throughput (requests/second)
rate(api_requests_total[5m])

# Error rate
rate(api_errors_total[5m]) / rate(api_requests_total[5m])
```

## Configuration

The model is loaded from: `results/classification/flat_model.pkl`

Make sure this file exists before starting the API.

## Next Steps

For production, consider:
- Authentication (API Keys, OAuth)
- Rate limiting
- Cloud Endpoints / API Gateway
- Structured logging
- Advanced health checks (readiness/liveness)
