#!/bin/bash
# Installation script with CUDA dependency handling

cd "$(dirname "$0")"

# Activate venv or create it
if [ ! -d "venv" ]; then
    echo "Creating venv..."
    python3 -m venv venv
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch CPU-only first to avoid CUDA issues
echo "Installing PyTorch (CPU-only)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "Installing other dependencies..."
pip install -r requirements.txt

echo "Installation completed!"
echo ""
echo "To activate venv: source venv/bin/activate"
