#!/bin/bash
# FastAPI startup script

cd "$(dirname "$0")"

# Activate venv
if [ ! -d "venv" ]; then
    echo "ERROR: venv does not exist. Run first: ./install.sh"
    exit 1
fi

source venv/bin/activate

# Check if model exists
MODEL_PATH="results/classification/flat_model.pkl"
if [ ! -f "$MODEL_PATH" ]; then
    echo "WARNING: Model does not exist yet."
    echo "   Run first: python3 src/train.py"
    echo ""
fi

# Start API
echo "Starting FastAPI..."
echo "   Documentation: http://localhost:8000/docs"
echo "   Metrics: http://localhost:8000/metrics"
echo ""

uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
