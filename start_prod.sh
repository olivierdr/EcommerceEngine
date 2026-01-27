#!/bin/bash
# Lance l'API (uvicorn prod) et le front (build + serve statique) en une commande.
# Utilise .env.production pour l'URL API.
# Ctrl+C arrête les deux.

cd "$(dirname "$0")"
ROOT="$PWD"

cleanup() {
  if [ -n "$UVICORN_PID" ]; then
    kill "$UVICORN_PID" 2>/dev/null
  fi
  if [ -n "$SERVE_PID" ]; then
    kill "$SERVE_PID" 2>/dev/null
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

# Vérifier .env.production
if [ ! -f "frontend-nextjs/.env.production" ]; then
  echo "⚠️  frontend-nextjs/.env.production absent. Utilisez .env.production.example comme base."
  echo ""
fi

FRONTEND_PORT=${FRONTEND_PORT:-3001}

echo "🚀 Démarrage PRODUCTION..."
echo "   API:       http://localhost:8000 (sans reload)"
echo "   Front:     Build puis serve statique sur http://localhost:${FRONTEND_PORT}"
echo "   Model:     GCS (MODEL_SOURCE=gcs)"
echo "   Ctrl+C pour tout arrêter."
echo ""

# Modèle depuis GCS en production
export MODEL_SOURCE=gcs
export MODEL_VERSION=${MODEL_VERSION:-v1.0.0}

# Lancer API en mode production (sans reload)
uvicorn src.api:app --host 0.0.0.0 --port 8000 &
UVICORN_PID=$!
sleep 2

# Build frontend avec .env.production
cd frontend-nextjs
echo "📦 Build du frontend (production)..."
npm run build

if [ ! -d "out" ]; then
  echo "❌ Erreur: le dossier 'out' n'existe pas après le build."
  echo "   Vérifiez que next.config.js a 'output: export'."
  exit 1
fi

# Servir les fichiers statiques
echo "🌐 Démarrage serveur statique sur http://localhost:${FRONTEND_PORT}..."
cd out

# Utiliser npx serve si disponible, sinon python http.server
if command -v npx &> /dev/null; then
  npx serve -p ${FRONTEND_PORT} -s . &
  SERVE_PID=$!
else
  echo "   Utilisation de Python http.server (installer 'serve' pour de meilleures performances)"
  python3 -m http.server ${FRONTEND_PORT} &
  SERVE_PID=$!
fi

# Attendre que l'utilisateur arrête
wait $SERVE_PID
