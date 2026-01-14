#!/bin/bash
# Script d'installation avec gestion des dépendances CUDA

cd "$(dirname "$0")"

# Activer le venv ou le créer
if [ ! -d "venv" ]; then
    echo "Création du venv..."
    python3 -m venv venv
fi

source venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip setuptools wheel

# Installer PyTorch CPU-only d'abord pour éviter les problèmes CUDA
echo "Installation de PyTorch (CPU-only)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Installer les autres dépendances
echo "Installation des autres dépendances..."
pip install -r requirements.txt

echo "✓ Installation terminée !"
echo ""
echo "Pour activer le venv : source venv/bin/activate"

