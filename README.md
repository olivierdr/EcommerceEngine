# E-commerce Classification

## Objective

Automatic classification of e-commerce products into leaf categories based on available information (title, description, brand, color). The project starts with a comprehensive audit of the existing taxonomy to understand its structure and identify potential inconsistencies, then explores a flat classification approach (baseline) with identification of uncertain products for human validation.

**For a detailed understanding of strategic choices, results and analyses, see the [SYNTHESIS.md](SYNTHESIS.md) file.**

## Project Structure

```
ClassificationEcommerce/
├── data/                          # Input data (CSV)
│   ├── .gitkeep
│   ├── trainset.csv              # Training data (30,520 products)
│   └── testset.csv               # Test data (7,631 products)
│
├── src/                           # Source code
│   ├── audit_taxonomy.py         # Step 1: Taxonomy audit
│   ├── train.py                  # Step 2: Classifier training
│   ├── evaluate.py               # Step 2: Evaluation and analyses
│   └── api.py                    # FastAPI with GCP observability
│
├── results/                       # Generated results
│   ├── audit/                    # Audit results
│   │   ├── category_names.json
│   │   ├── low_coherence_categories.json
│   │   └── high_coherence_categories.json
│   └── classification/           # Classification results
│       ├── flat_model.pkl
│       ├── certain_categories_analysis.json
│       ├── uncertain_categories_analysis.json
│       └── confusion_patterns.json
│
├── .cache/                        # Cache (embeddings, etc.)
├── requirements.txt
├── README.md                      # This file
└── SYNTHESIS.md                   # Detailed synthesis of approaches and results
```

## Installation

1. **Create a virtual environment** (recommended):
```bash
cd ./ClassificationEcommerce/
python3 -m venv venv
source venv/bin/activate 
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Important:** Place your `trainset.csv` and `testset.csv` files in the `data/` folder before running the scripts.

## Methodological Approach

### Step 1: Taxonomy Audit

Before any classification, a thorough analysis of the dataset allows to:
- Understand the hierarchical structure (depth, number of categories per level)
- Detect structural inconsistencies in `category_path`
- Evaluate semantic coherence of products within each category

**Execution:**
```bash
python3 src/audit_taxonomy.py
```

**Results:**
- 100 leaf categories identified
- Variable depth: 3 to 8 levels (median: 6)
- No structural inconsistencies detected
- 32 categories with low semantic coherence (< 0.4)
- 68 categories with high semantic coherence (>= 0.4)
- Category names generated from frequent keywords

**Generated files (in `results/audit/`):**
- `category_names.json`: Generated names for each category with examples
- `low_coherence_categories.json`: Categories with low semantic coherence
- `high_coherence_categories.json`: Categories with high semantic coherence

### Step 2: Flat Classification (Baseline)

**Principle:** Direct prediction of the leaf category among 100 classes, with identification of uncertain products based on confidence scores.

**Architecture:**
- Embeddings: Multilingual model `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Classifier: Logistic Regression
- Features: Concatenation of title + description

**Execution:**
```bash
python3 src/train.py      # Training
python3 src/evaluate.py   # Evaluation and analyses
```

**Results:**
- Accuracy: 77.47% on test set
- 75.5% of products with confidence >= 0.5 (certain)
- 24.5% of products with confidence < 0.5 (uncertain, require human validation)

**Generated files (in `results/classification/`):**
- `flat_model.pkl`: Saved trained model
- `certain_categories_analysis.json`: Top 10 categories with certain products
- `uncertain_categories_analysis.json`: Top 10 problematic categories
- `confusion_patterns.json`: Top 10 confusion patterns between categories

### Step 3: API with Observability (Optional)

**Principle:** REST API for production classification with OpenTelemetry instrumentation for Cloud Monitoring and Cloud Trace.

**Local execution:**
```bash
./start_api.sh
```

The API will be accessible at `http://localhost:8000` with:
- `/classify`: Classification endpoint (POST)
- `/health`: Health check
- `/metrics`: Prometheus metrics
- `/docs`: Swagger documentation

**Cloud Run deployment:**
```bash
export GOOGLE_CLOUD_PROJECT=your-project-id
./deploy_cloud_run.sh
```

**Observability:**
- Metrics exported to Cloud Monitoring (latency, throughput, errors, confidence)
- Traces exported to Cloud Trace (detailed spans for embedding, classification)
- Configurable dashboards and alerts (see [CLOUD_MONITORING.md](CLOUD_MONITORING.md))

## Technologies Used

- **Python 3**
- **scikit-learn**: Classification (Logistic Regression)
- **sentence-transformers**: Multilingual embeddings (FR/DE/EN)
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **FastAPI**: REST API (optional)
- **OpenTelemetry**: Observability (Cloud Monitoring, Cloud Trace)

## Additional Documentation

For detailed analysis of strategic choices, complete results, error examples and improvement axes, see the **[SYNTHESIS.md](SYNTHESIS.md)** file.
