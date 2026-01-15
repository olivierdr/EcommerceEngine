# API FastAPI - Classification E-commerce

API REST pour le service de classification de produits e-commerce avec instrumentation de base.

## 🚀 Démarrage rapide

### 1. Installer les dépendances

**Option A : Script automatique (recommandé)**
```bash
./install.sh
```

**Option B : Installation manuelle**
```bash
# Créer/activer le venv
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch CPU-only d'abord (évite les problèmes CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
pip install -r requirements.txt
```

> **Note** : PyTorch est installé en mode CPU-only pour éviter les erreurs avec `nvidia_cublas_cu12`. Si vous avez besoin du support GPU, installez PyTorch avec CUDA séparément.

### 2. Entraîner le modèle (si pas déjà fait)

```bash
python3 src/train.py
```

### 3. Démarrer l'API

```bash
./start_api.sh
```

Ou manuellement :

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

L'API sera accessible sur : http://localhost:8000

## 📋 Endpoints

### `POST /classify`
Classifie un produit dans une catégorie.

**Request:**
```json
{
  "title": "Samsung Galaxy S21",
  "description": "Smartphone Android avec écran 6.2 pouces"
}
```

**Response:**
```json
{
  "category_id": "12345",
  "category_path": "Electronique > Smartphones > Samsung",
  "confidence": 0.87,
  "processing_time_ms": 245.3
}
```

### `GET /health`
Vérifie l'état de santé de l'API.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `GET /metrics`
Endpoint Prometheus pour les métriques.

### `GET /docs`
Documentation interactive Swagger UI.

## 📊 Métriques (5 clés)

L'API expose 5 métriques principales via Prometheus :

1. **`api_request_duration_seconds`** (Histogram)
   - Latence des requêtes par endpoint et status

2. **`api_requests_total`** (Counter)
   - Throughput : nombre total de requêtes

3. **`api_errors_total`** (Counter)
   - Taux d'erreur : erreurs 4xx, 5xx, exceptions

4. **`api_confidence_score_average`** (Gauge)
   - Score de confiance moyen des prédictions

5. **`api_inference_duration_seconds`** (Histogram)
   - Temps d'inférence du modèle

## 🔍 Exemples d'utilisation

### Avec curl

```bash
# Classifier un produit
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "iPhone 14 Pro",
    "description": "Smartphone Apple avec puce A16"
  }'

# Vérifier la santé
curl http://localhost:8000/health

# Récupérer les métriques
curl http://localhost:8000/metrics
```

### Avec Python

```python
import requests

# Classifier un produit
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "title": "MacBook Pro 16",
        "description": "Ordinateur portable Apple M2"
    }
)
print(response.json())
```

## 📈 Visualisation des métriques

Les métriques sont au format Prometheus et peuvent être :
- Scrapées par Prometheus
- Visualisées dans Grafana
- Intégrées dans Cloud Monitoring (GCP)

### Exemple de requête PromQL

```promql
# Latence P95
histogram_quantile(0.95, api_request_duration_seconds_bucket)

# Throughput (requêtes/seconde)
rate(api_requests_total[5m])

# Taux d'erreur
rate(api_errors_total[5m]) / rate(api_requests_total[5m])
```

## 🛠️ Configuration

Le modèle est chargé depuis : `results/classification/flat_model.pkl`

Assurez-vous que ce fichier existe avant de démarrer l'API.

## 🔒 Prochaines étapes

Pour la production, considérer :
- Authentification (API Keys, OAuth)
- Rate limiting
- Cloud Endpoints / API Gateway
- Logging structuré
- Health checks avancés (readiness/liveness)

