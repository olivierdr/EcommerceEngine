"""
API FastAPI pour le service de classification E-commerce
Avec instrumentation de base (5 métriques clés)
"""

import time
import pickle
from pathlib import Path
from typing import Optional
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


# ==================== MODÈLE ====================

class ClassificationModel:
    """Wrapper pour charger et utiliser le modèle de classification"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or Path(__file__).parent.parent / 'results' / 'classification' / 'flat_model.pkl'
        self.classifier = None
        self.label_encoder = None
        self.embedding_model = None
        self.cat_to_path = {}
        self.load_model()
    
    def load_model(self):
        """Charge le modèle depuis le fichier pickle"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
        
        print(f"Chargement du modèle depuis {self.model_path}...")
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.cat_to_path = model_data.get('cat_to_path', {})
        
        # Charger le modèle d'embeddings
        embedding_model_name = model_data.get('embedding_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        print(f"Chargement du modèle d'embeddings: {embedding_model_name}...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        print("✓ Modèle chargé avec succès")
    
    def predict_single(self, title: str, description: str = ""):
        """Prédit la catégorie pour un seul produit"""
        # Préparer le texte
        text = f"{title} {description}".strip()
        
        # Générer l'embedding
        embedding = self.embedding_model.encode([text], show_progress_bar=False)
        
        # Prédire
        y_pred_encoded = self.classifier.predict(embedding)[0]
        y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]
        
        # Obtenir la probabilité
        y_proba = self.classifier.predict_proba(embedding)[0]
        confidence = float(np.max(y_proba))
        
        # Obtenir le chemin de catégorie
        category_path = self.cat_to_path.get(y_pred, "N/A")
        
        return {
            'category_id': str(y_pred),
            'category_path': category_path,
            'confidence': confidence
        }


# ==================== MÉTRIQUES ====================

# 1. Latence (temps de réponse)
request_duration = Histogram(
    'api_request_duration_seconds',
    'Temps de réponse des requêtes API',
    ['method', 'endpoint', 'status']
)

# 2. Throughput (nombre de requêtes)
requests_total = Counter(
    'api_requests_total',
    'Nombre total de requêtes',
    ['method', 'endpoint', 'status']
)

# 3. Taux d'erreur
errors_total = Counter(
    'api_errors_total',
    'Nombre total d\'erreurs',
    ['method', 'endpoint', 'error_type']
)

# 4. Score de confiance moyen
confidence_score_avg = Gauge(
    'api_confidence_score_average',
    'Score de confiance moyen des prédictions'
)

# 5. Temps d'inférence du modèle
inference_duration = Histogram(
    'api_inference_duration_seconds',
    'Temps d\'inférence du modèle de classification'
)

# Compteur pour calculer la moyenne de confiance
_confidence_sum = 0.0
_confidence_count = 0


# ==================== API ====================

app = FastAPI(
    title="E-commerce Classification API",
    description="API de classification de produits e-commerce avec observabilité",
    version="1.0.0"
)

# Charger le modèle au démarrage
try:
    model = ClassificationModel()
except Exception as e:
    print(f"ERREUR: Impossible de charger le modèle: {e}")
    model = None


# ==================== MODÈLES PYDANTIC ====================

class ProductRequest(BaseModel):
    title: str = Field(..., description="Titre du produit", min_length=1)
    description: Optional[str] = Field(default="", description="Description du produit")


class ClassificationResponse(BaseModel):
    category_id: str
    category_path: str
    confidence: float
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ==================== MIDDLEWARE POUR MÉTRIQUES ====================

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Middleware pour collecter les métriques"""
    start_time = time.time()
    method = request.method
    endpoint = request.url.path
    
    try:
        response = await call_next(request)
        status = response.status_code
        status_class = f"{status // 100}xx"
        
        # Enregistrer la latence
        duration = time.time() - start_time
        request_duration.labels(method=method, endpoint=endpoint, status=status_class).observe(duration)
        
        # Enregistrer le throughput
        requests_total.labels(method=method, endpoint=endpoint, status=status_class).inc()
        
        # Enregistrer les erreurs
        if status >= 400:
            error_type = "4xx" if status < 500 else "5xx"
            errors_total.labels(method=method, endpoint=endpoint, error_type=error_type).inc()
        
        return response
    
    except Exception as e:
        status = 500
        duration = time.time() - start_time
        
        # Enregistrer l'erreur
        errors_total.labels(method=method, endpoint=endpoint, error_type="exception").inc()
        request_duration.labels(method=method, endpoint=endpoint, status="5xx").observe(duration)
        requests_total.labels(method=method, endpoint=endpoint, status="5xx").inc()
        
        raise


# ==================== ENDPOINTS ====================

@app.post("/classify", response_model=ClassificationResponse)
async def classify_product(product: ProductRequest):
    """
    Classifie un produit dans une catégorie
    
    - **title**: Titre du produit (requis)
    - **description**: Description du produit (optionnel)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    try:
        # Mesurer le temps d'inférence
        inference_start = time.time()
        
        # Prédire
        result = model.predict_single(product.title, product.description or "")
        
        inference_time = time.time() - inference_start
        
        # Enregistrer le temps d'inférence
        inference_duration.observe(inference_time)
        
        # Mettre à jour la moyenne de confiance
        global _confidence_sum, _confidence_count
        _confidence_sum += result['confidence']
        _confidence_count += 1
        if _confidence_count > 0:
            confidence_score_avg.set(_confidence_sum / _confidence_count)
        
        return ClassificationResponse(
            category_id=result['category_id'],
            category_path=result['category_path'],
            confidence=result['confidence'],
            processing_time_ms=inference_time * 1000
        )
    
    except Exception as e:
        errors_total.labels(method="POST", endpoint="/classify", error_type="prediction_error").inc()
        raise HTTPException(status_code=500, detail=f"Erreur lors de la classification: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé de l'API"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None
    )


@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus pour les métriques"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Endpoint racine"""
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

