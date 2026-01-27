# E-commerce Classification

Complete system for automatic e-commerce product classification with REST API and interactive web interface. The project combines a machine learning model for classification, a FastAPI for service exposure, and a Next.js frontend for user interaction.

## Project Origin

This project was developed to address the need for automatic e-commerce product classification in a complex hierarchical taxonomy. The main objective is to improve product visibility and ensure a consistent user experience by automatically classifying products into the correct categories based on their textual information (title, description, brand, color).

The project addresses several identified needs:

- **Automatic classification**: Predict a product's leaf category among 100 possible categories from its textual information
- **REST API**: Expose the classification model via an API to enable integration into other systems
- **Interactive web interface**: Provide a user interface to test the model, load datasets, visualize results, and analyze performance
- **Model and data versioning**: Implement a versioning system to track the evolution of trained models and datasets used
- **Production deployment**: Deploy the API on Cloud Run and the frontend on Firebase Hosting for public availability

## Architecture

### Overview

The project follows a modular architecture with clear separation between backend (Python/FastAPI API) and frontend (Next.js/React), with integration of Google Cloud Platform services for storage and deployment.

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                       │
│  - Interactive user interface                               │
│  - Product testing                                          │
│  - Results visualization                                    │
│  - Deployed on Firebase Hosting                             │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP/REST
┌──────────────────────▼──────────────────────────────────────┐
│                    API (FastAPI)                            │
│  - /classify endpoint for classification                    │
│  - /testset endpoint to load dataset                        │
│  - /category-names endpoint for metadata                     │
│  - Prometheus metrics                                       │
│  - Deployed on Cloud Run                                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐ ┌────▼─────┐ ┌─────▼──────┐
│   ML Model   │ │  GCS      │ │  BigQuery  │
│  (Pickle)    │ │ (Storage) │ │ (Tracking) │
│              │ │           │ │            │
│ - flat_model │ │ - Models  │ │ - Metrics  │
│ - Embeddings │ │ - Datasets│ │ - Versions │
└──────────────┘ └───────────┘ └────────────┘
```

### Main Components

#### Backend (FastAPI)

- **ClassificationModel**: Wrapper class to load and use the classification model
  - Support for loading from local filesystem or Google Cloud Storage (GCS)
  - Automatic embedding management with sentence-transformers
  - Prediction with confidence scores

- **REST Endpoints**:
  - `POST /classify`: Classify a product (title + description)
  - `GET /testset`: Retrieve test dataset (CSV)
  - `GET /category-names`: Category metadata with examples
  - `GET /health`: Health check
  - `GET /metrics`: Prometheus metrics
  - `GET /docs`: Interactive Swagger documentation

- **Prometheus Metrics**:
  - Request latency (`api_request_duration_seconds`)
  - Throughput (`api_requests_total`)
  - Error rate (`api_errors_total`)
  - Average confidence score (`api_confidence_score_average`)
  - Inference time (`api_inference_duration_seconds`)

#### Frontend (Next.js)

- **Main Pages**:
  - `/`: Home page with product tester and results visualization
  - `/categories`: Browse all categories with examples

- **Components**:
  - `ProductTester`: Interface to test products individually or load a dataset
  - `ResultsTable`: Results table with True Category, Predicted Category, Confidence columns, etc.
  - `StatsCards`: Statistics cards (accuracy, average confidence, average latency)
  - `Charts`: Confidence distribution charts, accuracy by category, latency

- **Features**:
  - Load testset from API
  - Test random products (10 by default)
  - Manual testing of individual products
  - Save results in localStorage
  - Display readable category names (resolved from API)

#### Storage and Versioning

- **Google Cloud Storage (GCS)**:
  - Versioned model storage: `gs://bucket/models/{version}/model.pkl`
  - Versioned dataset storage: `gs://bucket/datasets/{version}/{split}set.csv`
  - Metadata storage: `gs://bucket/models/{version}/category_names.json`
  - Latest version cache: `gs://bucket/models/LATEST_VERSION.txt`

- **BigQuery**:
  - Model version tracking
  - Training and evaluation metrics
  - Performance history

- **Versioning**:
  - Version format: `v1.0.0`, `v1.1.0`, etc. (Semantic Versioning)
  - Each version includes the model, metadata, and associated datasets
  - Complete traceability via BigQuery
  - Support for `MODEL_VERSION=latest` to automatically point to the latest version
  - `LATEST_VERSION.txt` file in GCS for fast cache of the most recent version

### Environments

The project supports two operating modes:

- **Local mode** (`MODEL_SOURCE=local`):
  - Loads model from `results/classification/flat_model.pkl`
  - Loads categories from `results/audit/category_names.json`
  - Used for development and local testing

- **Production mode** (`MODEL_SOURCE=gcs`):
  - Loads model from Google Cloud Storage
  - Model version configured via `MODEL_VERSION` (default: `latest`)
  - `MODEL_VERSION=latest` automatically resolves to the latest available version in GCS
  - Used for Cloud Run deployments and production

## Project Structure

```
ClassificationEcommerce/
├── src/                          # Python source code
│   ├── api.py                    # Main FastAPI
│   ├── train.py                  # Training script with versioning
│   ├── evaluate.py               # Model evaluation
│   ├── audit_taxonomy.py         # Taxonomy audit
│   ├── data/                     # Data loading utilities
│   │   ├── loader.py             # Local/GCS loading
│   │   └── testset.csv           # Test dataset
│   ├── models/                   # Classification models
│   │   └── flat_classifier.py    # Flat classifier (baseline)
│   ├── training/                 # Training utilities
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── tracking/                 # Tracking and versioning
│   │   ├── gcs.py                # GCS upload/download
│   │   ├── bigquery.py           # BigQuery tracking
│   │   └── vertex_ai.py          # Vertex AI integration
│   └── utils/                    # Utilities
│       ├── config.py             # GCP configuration
│       └── category_names.py     # Category name management
│
├── frontend-nextjs/              # Next.js frontend
│   ├── app/                      # App Router pages
│   │   ├── page.tsx              # Home page
│   │   ├── categories/           # Categories page
│   │   ├── config.ts             # API URL configuration
│   │   └── types.ts              # TypeScript types
│   ├── components/               # React components
│   │   ├── ProductTester.tsx
│   │   ├── ResultsTable.tsx
│   │   ├── StatsCards.tsx
│   │   └── Charts.tsx
│   ├── scripts/                  # Test scripts
│   │   ├── test-api-flow.mjs
│   │   └── test-parse-csv.mjs
│   ├── .env.production           # Production configuration
│   └── firebase.json             # Firebase Hosting configuration
│
├── results/                      # Generated results (local)
│   ├── classification/
│   │   └── flat_model.pkl        # Trained model
│   └── audit/
│       └── category_names.json   # Category names
│
├── scripts/                      # Deployment and management scripts
│   └── update_latest_version.sh  # Manual LATEST_VERSION.txt update
│
├── start_local.sh                # Local startup (API + Frontend)
├── start_prod.sh                 # Local production test
├── deploy_cloud_run.sh           # Cloud Run API deployment
├── deploy_all.sh                 # Complete deployment (API + Frontend)
├── Dockerfile                    # Docker image for Cloud Run
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Technologies Used

### Backend

- **Python 3.11**: Main language
- **FastAPI**: Web framework for REST API
- **scikit-learn**: Classification (Logistic Regression)
- **sentence-transformers**: Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **prometheus-client**: Prometheus metrics
- **google-cloud-storage**: GCS integration
- **google-cloud-bigquery**: BigQuery tracking

### Frontend

- **Next.js 14**: React framework with App Router
- **TypeScript**: Static typing
- **Tailwind CSS**: Styling
- **Recharts**: Data visualization
- **Firebase Hosting**: Frontend deployment

### Infrastructure

- **Google Cloud Run**: API hosting (serverless)
- **Google Cloud Storage**: Model and dataset storage
- **Google BigQuery**: Tracking and metrics
- **Firebase Hosting**: Static frontend hosting
- **Docker**: Containerization for Cloud Run

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Google Cloud SDK (`gcloud`)
- Firebase CLI (`firebase-tools`)
- Configured Google Cloud Platform account

### Dependency Installation

**Backend:**

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch CPU-only (recommended to avoid CUDA issues)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

**Frontend:**

```bash
cd frontend-nextjs
npm install
```

### Google Cloud Configuration

```bash
# Authentication
gcloud auth login

# Set default project
gcloud config set project master-ai-cloud

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable bigquery.googleapis.com
```

### Firebase Configuration

```bash
cd frontend-nextjs
firebase login
firebase init hosting
```

## Usage

### Local Development

**Option 1: Automatic script (recommended)**

```bash
./start_local.sh
```

This script launches:
- API on `http://localhost:8000` (development mode with reload)
- Frontend on `http://localhost:3000` (Next.js development mode)
- Loads model from local filesystem

**Option 2: Manual launch**

```bash
# Terminal 1: API
source venv/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Frontend
cd frontend-nextjs
npm run dev
```

### Local Production Test

To test production behavior (model from GCS):

```bash
./start_prod.sh
```

This script:
- Launches API on `http://localhost:8000` (without reload, production mode)
- Builds frontend with `.env.production`
- Serves static files on `http://localhost:3001`
- Loads model from GCS (`MODEL_SOURCE=gcs`)

### Model Training

**Local training:**

```bash
python3 src/train.py --version v1.0.6 --local-only
```

**Training with GCS/BigQuery upload:**

```bash
python3 src/train.py --version v1.0.6
```

The script:
- Trains model on training dataset
- Evaluates on test dataset
- Saves model locally in `results/classification/`
- Uploads to GCS in `gs://bucket/models/{version}/model.pkl`
- Automatically updates `LATEST_VERSION.txt` in GCS with the new version
- Records metrics in BigQuery

### Taxonomy Audit

```bash
python3 src/audit_taxonomy.py
```

Generates:
- Hierarchical structure analysis
- Inconsistency detection
- Semantic coherence evaluation
- Category name generation (`results/audit/category_names.json`)

### Deployment

**Complete deployment (API + Frontend):**

```bash
./deploy_all.sh
```

This script:
1. Verifies Google Cloud and Firebase authentications
2. Deploys API on Cloud Run with `MODEL_VERSION=latest` (automatically resolves to latest version)
3. Retrieves deployed API URL
4. Updates `frontend-nextjs/.env.production` with API URL
5. Builds and deploys frontend on Firebase Hosting

**With specific version:**

```bash
MODEL_VERSION=v1.0.5 ./deploy_all.sh
```

**API deployment only:**

```bash
./deploy_cloud_run.sh
```

**Frontend deployment only:**

```bash
cd frontend-nextjs
npm run deploy:firebase
```

## Configuration

### Environment Variables

**API (Backend):**

- `MODEL_SOURCE`: Model source (`local` or `gcs`, default: `local`)
- `MODEL_VERSION`: Model version to load from GCS (default: `latest`)
  - `latest`: Automatically resolves to the latest available version in GCS
  - `v1.0.5`, `v1.0.6`, etc.: Specific version
- `GOOGLE_CLOUD_PROJECT`: GCP project ID (default: `master-ai-cloud`)
- `GCS_BUCKET`: GCS bucket name (default: `master-ai-cloud-ecommerce-ml`)

**Frontend:**

- `NEXT_PUBLIC_API_URL`: API URL (defined in `.env.production` for build)

### Configuration Files

- `src/utils/config.py`: GCP configuration (PROJECT_ID, REGION, BUCKET, etc.)
- `frontend-nextjs/.env.production`: Frontend production configuration
- `Dockerfile`: Docker image configuration for Cloud Run
- `firebase.json`: Firebase Hosting configuration

## Model Version Management

### Automatic Versioning System

The project uses Semantic Versioning (`v1.0.0`, `v1.0.1`, `v1.1.0`, etc.) with automatic support for the "latest" version.

**How it works:**

1. **During training**:
   ```bash
   python3 src/train.py --version v1.0.6
   ```
   - Uploads model to `gs://bucket/models/v1.0.6/model.pkl`
   - Automatically updates `gs://bucket/models/LATEST_VERSION.txt` with `v1.0.6`

2. **During deployment with `latest`**:
   ```bash
   ./deploy_all.sh  # Uses MODEL_VERSION=latest by default
   ```
   - API first reads `LATEST_VERSION.txt` to know the latest version
   - If file doesn't exist, lists all versions in GCS and finds the most recent numerically
   - Automatically loads this version

3. **Manual update of LATEST_VERSION.txt**:
   ```bash
   ./scripts/update_latest_version.sh v1.0.5
   ```
   Useful for rollback to an older version or initialization.

**Usage examples:**

```bash
# Deploy with latest version (automatic)
./deploy_all.sh

# Deploy with specific version
MODEL_VERSION=v1.0.5 ./deploy_all.sh

# Test a version locally (production)
MODEL_VERSION=v1.0.3 ./start_prod.sh

# Train a new version (automatically updates LATEST_VERSION.txt)
python3 src/train.py --version v1.0.7
```

## Performance and Metrics

### Model Metrics

- **Overall accuracy**: 77.47% on test set
- **Confidence distribution**: 75.5% of products with confidence >= 0.5
- **Uncertain products**: 24.5% require human validation

### API Metrics (Prometheus)

Metrics are exposed on `/metrics`:

- `api_request_duration_seconds`: Request latency
- `api_requests_total`: Total number of requests
- `api_errors_total`: Number of errors
- `api_confidence_score_average`: Average confidence score
- `api_inference_duration_seconds`: Model inference time

### Visualization

Metrics can be:
- Scraped by Prometheus
- Visualized in Grafana
- Integrated into Google Cloud Monitoring

## Additional Documentation

- **API_README.md**: Detailed API documentation
- **SYNTHESIS.md**: Methodological synthesis and detailed results
- **ARCHITECTURE.md**: Detailed system architecture
- **CLOUD_MONITORING.md**: Cloud monitoring guide

## Future Improvements

### Technical

- Migration to more sophisticated models (fine-tuned BERT, Transformers)
- Hierarchy exploitation (top-down or hybrid approach)
- Efficient brand and color integration
- Adaptive confidence threshold per category

### Operational

- Automatic new category detection
- Real-time performance monitoring
- Human correction integration workflow
- Active learning to prioritize products to validate

## License

This project is a demonstration and learning project.
