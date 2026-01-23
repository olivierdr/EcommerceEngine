"""
Vertex AI Experiments tracking
"""
import sys
from datetime import datetime
from pathlib import Path
from google.cloud import aiplatform

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import PROJECT_ID, REGION, EXPERIMENT_NAME


def init_vertex_ai(experiment_name=None):
    """Initialize Vertex AI"""
    if experiment_name is None:
        experiment_name = EXPERIMENT_NAME
    
    aiplatform.init(project=PROJECT_ID, location=REGION, experiment=experiment_name)
    return experiment_name


def start_run(run_id=None):
    """Start a new run in Vertex AI"""
    if run_id is None:
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    aiplatform.start_run(run=run_id)
    return run_id


def log_params(params):
    """Log hyperparameters to Vertex AI"""
    aiplatform.log_params(params)


def log_metrics(metrics):
    """Log metrics to Vertex AI"""
    aiplatform.log_metrics(metrics)


def end_run():
    """End the current run"""
    aiplatform.end_run()

