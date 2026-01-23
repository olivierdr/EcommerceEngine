"""
Data loading utilities (local or GCS)
"""
import sys
import pandas as pd
from pathlib import Path
from google.cloud import storage

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import GCS_BUCKET, PROJECT_ID


def load_dataset_from_local(dataset_version, base_path=None):
    """Load dataset from local filesystem"""
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent
    
    train_path = base_path / 'data' / 'trainset.csv'
    val_path = base_path / 'data' / 'valset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    datasets = {}
    if train_path.exists():
        datasets['train'] = pd.read_csv(train_path)
    if val_path.exists():
        datasets['val'] = pd.read_csv(val_path)
    if test_path.exists():
        datasets['test'] = pd.read_csv(test_path)
    
    return datasets


def load_dataset_from_gcs(dataset_version, bucket_name=None):
    """Load dataset from GCS"""
    if bucket_name is None:
        bucket_name = GCS_BUCKET
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(bucket_name)
    
    datasets = {}
    
    # Try to load train, val, test
    for split in ['train', 'val', 'test']:
        gcs_path = f"datasets/{dataset_version}/{split}set.csv"
        blob = bucket.blob(gcs_path)
        
        if blob.exists():
            # Download to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as tmp_file:
                blob.download_to_filename(tmp_file.name)
                datasets[split] = pd.read_csv(tmp_file.name)
                import os
                os.unlink(tmp_file.name)
        else:
            print(f"   Warning: {gcs_path} not found in GCS")
    
    return datasets


def load_dataset(dataset_version, local_only=False, base_path=None, bucket_name=None):
    """Load dataset from local or GCS based on local_only flag"""
    if local_only:
        print(f"Loading dataset {dataset_version} from local filesystem...")
        return load_dataset_from_local(dataset_version, base_path)
    else:
        print(f"Loading dataset {dataset_version} from GCS...")
        try:
            return load_dataset_from_gcs(dataset_version, bucket_name)
        except Exception as e:
            print(f"Warning: Could not load from GCS: {e}")
            print("Falling back to local filesystem...")
            return load_dataset_from_local(dataset_version, base_path)

