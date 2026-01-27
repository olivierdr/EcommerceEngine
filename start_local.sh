#!/bin/bash
# Lance l'API (uvicorn) et le front (npm run dev) en une commande.
# Ctrl+C arrête les deux.

cd "$(dirname "$0")"
ROOT="$PWD"

cleanup() {
  if [ -n "$UVICORN_PID" ]; then
    kill "$UVICORN_PID" 2>/dev/null
  fi
  exit 0
}
trap cleanup EXIT INT TERM

# Venv pour l'API
if [ -d "venv" ]; then
  source venv/bin/activate
else
  echo "⚠️  Pas de venv trouvé. L'API peut échouer si uvicorn n'est pas installé."
fi

# Optionnel : avertissement si pas de modèle
if [ ! -f "results/classification/flat_model.pkl" ]; then
  echo "⚠️  results/classification/flat_model.pkl absent. API peut renvoyer 503."
  echo ""
fi

echo "🚀 Démarrage API (port 8000) + front (port 3000)..."
echo "   API:       http://localhost:8000"
echo "   Front:     http://localhost:3000"
echo "   Model:     Local (results/classification/flat_model.pkl)"
echo "   Ctrl+C pour tout arrêter."
echo ""

# Modèle local par défaut (pas de MODEL_SOURCE ou MODEL_SOURCE=local)
export MODEL_SOURCE=local

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload &
UVICORN_PID=$!
sleep 2
cd frontend-nextjs && npm run dev
