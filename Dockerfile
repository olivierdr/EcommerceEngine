# Dockerfile pour Cloud Run
FROM python:3.11-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY src/ ./src/

# Créer le répertoire pour le modèle et copier le modèle
RUN mkdir -p ./results/classification
COPY results/classification/flat_model.pkl ./results/classification/flat_model.pkl

# Exposer le port (Cloud Run utilise PORT env var)
ENV PORT=8080
EXPOSE 8080

# Commande de démarrage
CMD exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1

