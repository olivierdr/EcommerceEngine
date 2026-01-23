"""
Configuration constants
"""
import os

# GCP Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "master-ai-cloud")
REGION = os.getenv("GCP_REGION", "europe-west1")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "ecommerce-classification-v1")
GCS_BUCKET = os.getenv("GCS_BUCKET", "master-ai-cloud-ecommerce-ml")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "Ecommerce")

