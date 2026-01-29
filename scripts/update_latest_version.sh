#!/bin/bash
# Met à jour LATEST_VERSION.txt dans GCS avec une version spécifique
# Usage: ./scripts/update_latest_version.sh v1.0.5

set -e

VERSION=${1:-}

if [ -z "${VERSION}" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 v1.0.5"
    exit 1
fi

# Vérifier le format de version
if [[ ! "${VERSION}" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Version must match format vX.Y.Z (e.g., v1.0.5)"
    exit 1
fi

cd "$(dirname "$0")/.."

# Vérifier que gcloud est disponible
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed"
    exit 1
fi

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-"master-ai-cloud"}
BUCKET=${GCS_BUCKET:-"master-ai-cloud-ecommerce-ml"}

echo "Mise à jour LATEST_VERSION.txt..."
echo "  Bucket: ${BUCKET}"
echo "  Version: ${VERSION}"
echo ""

# Vérifier que la version existe dans GCS
if ! gsutil ls "gs://${BUCKET}/models/${VERSION}/model.pkl" &>/dev/null; then
    echo "Error: Version ${VERSION} not found in GCS"
    echo "  Check: gsutil ls gs://${BUCKET}/models/${VERSION}/"
    exit 1
fi

# Mettre à jour LATEST_VERSION.txt
echo "${VERSION}" | gsutil cp - "gs://${BUCKET}/models/LATEST_VERSION.txt"

echo "✓ LATEST_VERSION.txt mis à jour avec ${VERSION}"
echo "  Path: gs://${BUCKET}/models/LATEST_VERSION.txt"
