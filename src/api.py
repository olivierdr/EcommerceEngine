"""
FastAPI for E-commerce Classification Service
With OpenTelemetry instrumentation for Cloud Monitoring and Cloud Trace
"""

import os
import time
import pickle
from pathlib import Path
from typing import Optional
from contextlib import nullcontext
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.gcp.trace import CloudTraceSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    
    # Try to import Cloud Monitoring (may be in alpha)
    try:
        from opentelemetry.exporter.gcp.monitoring import GcpMonitoringMetricsExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        GCP_MONITORING_AVAILABLE = True
    except ImportError:
        GCP_MONITORING_AVAILABLE = False
        print("Cloud Monitoring exporter not available, only traces will be exported")
    
    # Initialize OpenTelemetry for GCP (only if on GCP)
    if os.getenv("GOOGLE_CLOUD_PROJECT"):
        # Configure Resource with service metadata
        resource = Resource.create({
            "service.name": "ecommerce-classification-api",
            "service.version": "1.0.0",
        })
        
        # Configure Cloud Trace
        trace_provider = TracerProvider(resource=resource)
        cloud_trace_exporter = CloudTraceSpanExporter()
        trace_provider.add_span_processor(BatchSpanProcessor(cloud_trace_exporter))
        trace.set_tracer_provider(trace_provider)
        print("Cloud Trace configured")
        
        # Configure Cloud Monitoring (metrics) if available
        if GCP_MONITORING_AVAILABLE:
            try:
                metric_reader = PeriodicExportingMetricReader(
                    GcpMonitoringMetricsExporter(),
                    export_interval_millis=60000  # Export every 60 seconds
                )
                metrics_provider = MeterProvider(
                    resource=resource,
                    metric_readers=[metric_reader]
                )
                metrics.set_meter_provider(metrics_provider)
                print("Cloud Monitoring configured")
            except Exception as e:
                print(f"Error configuring Cloud Monitoring: {e}")
        
        OTEL_ENABLED = True
    else:
        OTEL_ENABLED = False
except ImportError as e:
    OTEL_ENABLED = False
    trace = None
    metrics = None
    print(f"OpenTelemetry not available: {e}")


# ==================== MODEL ====================

class ClassificationModel:
    """Wrapper for loading and using the classification model"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(__file__).parent.parent / 'results' / 'classification' / 'flat_model.pkl'
        self.classifier = None
        self.label_encoder = None
        self.embedding_model = None
        self.cat_to_path = {}
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
        
        # Load embedding model
        embedding_model_name = model_data.get('embedding_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"Loading embedding model: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print("Model loaded successfully")
    
    def predict_single(self, title: str, description: str = ""):
        """Predict category for a single product"""
        tracer = trace.get_tracer(__name__) if OTEL_ENABLED else None
        
        # Create main span if OpenTelemetry is enabled
        span_ctx = tracer.start_as_current_span("model.predict") if tracer else nullcontext()
        
        with span_ctx:
            # Prepare text
            text = f"{title} {description}".strip()
            
            # Generate embedding (with span)
            if tracer:
                with tracer.start_as_current_span("model.embedding"):
                    embedding = self.embedding_model.encode([text], show_progress_bar=False)
            else:
                embedding = self.embedding_model.encode([text], show_progress_bar=False)
            
            # Predict (with span)
            if tracer:
                with tracer.start_as_current_span("model.classify"):
                    y_pred_encoded = self.classifier.predict(embedding)[0]
                    y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
                    y_proba = self.classifier.predict_proba(embedding)[0]
            else:
                y_pred_encoded = self.classifier.predict(embedding)[0]
                y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
                y_proba = self.classifier.predict_proba(embedding)[0]
            
            confidence = float(np.max(y_proba))
            category_path = self.cat_to_path.get(y_pred, "N/A")
            
            # Add attributes to span
            if tracer:
                span = trace.get_current_span()
                if span:
                    span.set_attribute("prediction.category_id", str(y_pred))
                    span.set_attribute("prediction.confidence", confidence)
            
            return {
                'category_id': str(y_pred),
                'category_path': category_path,
                'confidence': confidence
            }


# ==================== METRICS ====================

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

# Instrument FastAPI with OpenTelemetry (if available)
if OTEL_ENABLED:
    FastAPIInstrumentor.instrument_app(app)
    RequestsInstrumentor().instrument()

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
        status = 500
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
    
    tracer = trace.get_tracer(__name__) if OTEL_ENABLED else None
    
    try:
        # Measure inference time
        inference_start = time.time()
        
        # Predict (span will be created in predict_single)
        result = model.predict_single(product.title, product.description or "")
        
        inference_time = time.time() - inference_start
        
        # Record inference time
        inference_duration.observe(inference_time)
        
        # Update average confidence
        global _confidence_sum, _confidence_count
        _confidence_sum += result['confidence']
        _confidence_count += 1
        if _confidence_count > 0:
            confidence_score_avg.set(_confidence_sum / _confidence_count)
        
        # Add attributes to main span
        if tracer:
            span = trace.get_current_span()
            if span:
                span.set_attribute("response.confidence", result['confidence'])
                span.set_attribute("response.processing_time_ms", inference_time * 1000)
        
        return ClassificationResponse(
            category_id=result['category_id'],
            category_path=result['category_path'],
            confidence=result['confidence'],
            processing_time_ms=inference_time * 1000
        )
    
    except Exception as e:
        errors_total.labels(method="POST", endpoint="/classify", error_type="prediction_error").inc()
        
        # Record error in span
        if tracer:
            span = trace.get_current_span()
            if span:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.get("/metrics")
async def metrics():
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
