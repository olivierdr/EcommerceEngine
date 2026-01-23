"""
GCS upload utilities
"""
import sys
import json
from pathlib import Path
from datetime import datetime
from google.cloud import storage

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import GCS_BUCKET, PROJECT_ID


def upload_file_to_gcs(local_path, gcs_path, bucket_name=None):
    """Upload a single file to GCS"""
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{gcs_path}"


def upload_model(model_path, version, bucket_name=None):
    """Upload model to GCS"""
    gcs_model_path = f"models/{version}/model.pkl"
    return upload_file_to_gcs(model_path, gcs_model_path, bucket_name)


def upload_metadata(metadata, version, bucket_name=None, base_path=None):
    """Upload metadata JSON to GCS"""
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent / 'results' / 'classification'
    
    # Save metadata locally first
    metadata_path = base_path / 'metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Upload to GCS
    gcs_metadata_path = f"models/{version}/metadata.json"
    return upload_file_to_gcs(metadata_path, gcs_metadata_path, bucket_name)


def download_model(version, local_path, bucket_name=None):
    """Download model from GCS"""
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"models/{version}/model.pkl")
    
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local_path))
    return local_path

