"""
FastAPI for E-commerce Classification Service
With structured logging for Cloud Logging metrics
"""

import os
import time
import pickle
import logging
import json
from pathlib import Path
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Configure structured logging for Cloud Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# ==================== MODEL ====================

class ClassificationModel:
    """Wrapper for loading and using the classification model"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(__file__).parent.parent / 'results' / 'classification' / 'flat_model.pkl'
        self.classifier = None
        self.label_encoder = None
        self.embedding_model = None
        self.cat_to_path = {}
        self.cat_to_name = {}
        self.load_model()
    
    def load_model(self):
        """Load model from pickle file"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.cat_to_path = model_data.get('cat_to_path', {})
        
        # Load category names
        names_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cat_id, info in data.items():
                    self.cat_to_name[cat_id] = info['name'] if isinstance(info, dict) else info
            print(f"Loaded {len(self.cat_to_name)} category names")
        
        # Load embedding model
        embedding_model_name = model_data.get('embedding_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print("Model loaded successfully")
    
    def predict_single(self, title: str, description: str = ""):
        """Predict category for a single product"""
        # Prepare text
        text = f"{title} {description}".strip()
        
        # Generate embedding
        embedding = self.embedding_model.encode([text], show_progress_bar=False)
        
        # Predict
        y_pred_encoded = self.classifier.predict(embedding)[0]
        y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
        y_proba = self.classifier.predict_proba(embedding)[0]
        
        confidence = float(np.max(y_proba))
        category_path = self.cat_to_path.get(y_pred, "N/A")
        category_name = self.cat_to_name.get(y_pred, "Unknown")
        
        return {
            'category_id': str(y_pred),
            'category_name': category_name,
            'category_path': category_path,
            'confidence': confidence
        }


# ==================== PROMETHEUS METRICS ====================

# 1. Latency (response time)
request_duration = Histogram(
    'api_request_duration_seconds',
    'API request response time',
    ['method', 'endpoint', 'status']
)

# 2. Throughput (request count)
requests_total = Counter(
    'api_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

# 3. Error rate
errors_total = Counter(
    'api_errors_total',
    'Total number of errors',
    ['method', 'endpoint', 'error_type']
)

# 4. Average confidence score
confidence_score_avg = Gauge(
    'api_confidence_score_average',
    'Average confidence score of predictions'
)

# 5. Model inference time
inference_duration = Histogram(
    'api_inference_duration_seconds',
    'Model classification inference time'
)

# Counter for calculating average confidence
_confidence_sum = 0.0
_confidence_count = 0


# ==================== API ====================

app = FastAPI(
    title="E-commerce Classification API",
    description="Product classification API with observability",
    version="1.0.0"
)

# Load model at startup
try:
    model = ClassificationModel()
except Exception as e:
    print(f"ERROR: Unable to load model: {e}")
    model = None


# ==================== PYDANTIC MODELS ====================

class ProductRequest(BaseModel):
    title: str = Field(..., description="Product title", min_length=1)
    description: Optional[str] = Field(default="", description="Product description")


class ClassificationResponse(BaseModel):
    category_id: str
    category_name: str
    category_path: str
    confidence: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ==================== MIDDLEWARE FOR METRICS ====================

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware for collecting metrics"""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        status = response.status_code
        status_class = f"{status // 100}xx"
        
        # Record latency
        duration = time.time() - start_time
        request_duration.labels(method=method, endpoint=endpoint, status=status_class).observe(duration)
        
        # Record throughput
        requests_total.labels(method=method, endpoint=endpoint, status=status_class).inc()
        
        # Record errors
        if status >= 400:
            error_type = "4xx" if status < 500 else "5xx"
            errors_total.labels(method=method, endpoint=endpoint, error_type=error_type).inc()
        
        return response
    
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error
        errors_total.labels(method=method, endpoint=endpoint, error_type="exception").inc()
        request_duration.labels(method=method, endpoint=endpoint, status="5xx").observe(duration)
        requests_total.labels(method=method, endpoint=endpoint, status="5xx").inc()
        
        raise


# ==================== ENDPOINTS ====================

@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(product: ProductRequest):
    """
    Classify a product into a category
    
    - **title**: Product title (required)
    - **description**: Product description (optional)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Measure inference time
        inference_start = time.time()
        
        # Predict
        result = model.predict_single(product.title, product.description or "")
        
        inference_time = time.time() - inference_start
        
        # Record inference time (Prometheus)
        inference_duration.observe(inference_time)
        
        # Update average confidence (Prometheus)
        global _confidence_sum, _confidence_count
        _confidence_sum += result['confidence']
        _confidence_count += 1
        if _confidence_count > 0:
            confidence_score_avg.set(_confidence_sum / _confidence_count)
        
        # Structured logging for Cloud Logging metrics
        is_uncertain = result['confidence'] < 0.5
        log_entry = {
            "severity": "INFO",
            "message": "classification_result",
            "category_id": result['category_id'],
            "category_name": result['category_name'],
            "confidence": round(result['confidence'], 3),
            "is_uncertain": is_uncertain,
            "inference_time_ms": round(inference_time * 1000, 1),
            "title_length": len(product.title)
        }
        logger.info(json.dumps(log_entry))
        
        return ClassificationResponse(
            category_id=result['category_id'],
            category_name=result['category_name'],
            category_path=result['category_path'],
            confidence=result['confidence'],
            processing_time_ms=inference_time * 1000
        )
    
    except Exception as e:
        errors_total.labels(method="POST", endpoint="/classify", error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.get("/metrics")
async def get_metrics():
    """Prometheus endpoint for metrics"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "E-commerce Classification API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "/classify (POST)",
            "health": "/health (GET)",
            "metrics": "/metrics (GET)",
            "docs": "/docs"
        }
    }
