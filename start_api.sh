#!/bin/bash
# Script de démarrage de l'API FastAPI

cd "$(dirname "$0")"

# Activer le venv
if [ ! -d "venv" ]; then
    echo "❌ Le venv n'existe pas. Exécutez d'abord: ./install.sh"
    exit 1
fi

source venv/bin/activate

# Vérifier que le modèle existe
MODEL_PATH="results/classification/flat_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Attention: Le modèle n'existe pas encore."
    echo "   Exécutez d'abord: python3 src/classify_flat.py"
    echo ""
fi

# Démarrer l'API
echo "🚀 Démarrage de l'API FastAPI..."
echo "   Documentation: http://localhost:8000/docs"
echo "   Métriques: http://localhost:8000/metrics"
echo ""

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

