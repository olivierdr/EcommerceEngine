# Dockerfile pour Cloud Run
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model (cached in image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Copy source code
COPY src/ ./src/

# Create directories and copy model, category names, and testset
RUN mkdir -p ./results/classification ./results/audit ./data
COPY results/classification/flat_model.pkl ./results/classification/flat_model.pkl
COPY results/audit/category_names.json ./results/audit/category_names.json
COPY data/testset.csv ./data/testset.csv

# Exposer le port (Cloud Run utilise PORT env var)
ENV PORT=8080
EXPOSE 8080

# Commande de démarrage
CMD exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

