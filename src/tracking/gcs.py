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


def get_latest_version(bucket_name=None):
    """Get the latest model version from GCS.
    First tries to read LATEST_VERSION.txt, then falls back to listing and comparing versions.
    """
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    
    # Try to read LATEST_VERSION.txt first (faster)
    latest_blob = bucket.blob("models/LATEST_VERSION.txt")
    if latest_blob.exists():
        try:
            latest_version = latest_blob.download_as_text().strip()
            if latest_version:
                return latest_version
        except Exception as e:
            print(f"Warning: Could not read LATEST_VERSION.txt: {e}")
            print("Falling back to listing versions...")
    
    # Fallback: list all versions and find the latest
    import re
    versions = set()
    
    # List blobs with prefix "models/v" and extract unique version strings
    for blob in bucket.list_blobs(prefix="models/v", delimiter="/"):
        # Extract version from path like "models/v1.0.5/model.pkl" or "models/v1.0.5/"
        match = re.match(r"models/(v\d+\.\d+\.\d+)", blob.name)
        if match:
            version_str = match.group(1)
            versions.add(version_str)
    
    # Also check prefixes (folders) directly
    for prefix in bucket.list_blobs(prefix="models/v", delimiter="/"):
        # For delimiter listing, check if it's a folder
        if prefix.name.endswith("/"):
            match = re.match(r"models/(v\d+\.\d+\.\d+)/", prefix.name)
            if match:
                versions.add(match.group(1))
    
    if not versions:
        raise ValueError("No model versions found in GCS")
    
    # Sort versions numerically (v1.0.5 > v1.0.1)
    def version_key(v):
        parts = v[1:].split('.')  # Remove 'v' and split
        return tuple(int(p) for p in parts)
    
    latest = max(versions, key=version_key)
    print(f"Found {len(versions)} version(s) in GCS, latest is {latest}")
    return latest


def update_latest_version(version, bucket_name=None):
    """Update LATEST_VERSION.txt in GCS with the given version"""
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("models/LATEST_VERSION.txt")
    blob.upload_from_string(version)
    return f"gs://{bucket_name}/models/LATEST_VERSION.txt"

