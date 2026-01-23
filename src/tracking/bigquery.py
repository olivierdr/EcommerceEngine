"""
BigQuery tracking utilities
"""
import sys
import json
import os
from datetime import datetime
from pathlib import Path
from google.cloud import bigquery

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.config import PROJECT_ID, BIGQUERY_DATASET, EXPERIMENT_NAME


def save_training_run(run_id, model_version, dataset_version, metrics, hyperparameters,
                     training_time, model_size_mb, model_gcs_path):
    """Save training run metadata to BigQuery training_runs table"""
    client = bigquery.Client(project=PROJECT_ID)
    table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.training_runs"
    now = datetime.now()
    
    row = {
        "run_id": run_id,
        "experiment_name": EXPERIMENT_NAME,
        "vertex_ai_run_id": run_id,
        "created_at": now.isoformat(),
        "completed_at": now.isoformat(),
        "duration_seconds": training_time,
        "model_version": model_version,
        "algorithm": "LogisticRegression",
        "model_path": model_gcs_path,
        "hyperparameters": json.dumps(hyperparameters),
        "dataset_version": dataset_version,
        "train_samples": metrics.get("train_samples", 0),
        "val_samples": metrics.get("val_samples", 0),
        "test_samples": metrics.get("test_samples", 0),
        "train_accuracy": metrics.get("train_accuracy"),
        "val_accuracy": metrics.get("val_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
        "train_f1_score": metrics.get("train_f1"),
        "val_f1_score": metrics.get("val_f1"),
        "test_f1_score": metrics.get("test_f1"),
        "avg_inference_time_ms": metrics.get("avg_inference_time_ms"),
        "p95_inference_time_ms": metrics.get("p95_inference_time_ms"),
        "model_size_mb": model_size_mb,
        "status": "completed",
        "is_production": False,
        "is_best": False,
        "git_commit_hash": os.environ.get("GIT_COMMIT", ""),
        "notes": f"Automated training run {run_id}",
        "error_message": None
    }
    
    errors = client.insert_rows_json(table, [row])
    if errors:
        raise Exception(f"BigQuery insert errors: {errors}")
    return True


def save_model_version(run_id, model_version, model_gcs_path, model_size_mb, 
                      hyperparameters, metrics):
    """Save model version to BigQuery model_versions table"""
    client = bigquery.Client(project=PROJECT_ID)
    table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.model_versions"
    now = datetime.now()
    
    row = {
        "model_version": model_version,
        "run_id": run_id,
        "algorithm": "LogisticRegression",
        "model_path": model_gcs_path,
        "model_size_mb": model_size_mb,
        "embeddings_model": hyperparameters.get("embedding_model", ""),
        "test_accuracy": metrics.get("val_accuracy"),  # Using val as test for now
        "test_f1_score": metrics.get("val_f1"),
        "avg_confidence": metrics.get("avg_confidence"),
        "avg_inference_time_ms": metrics.get("avg_inference_time_ms"),
        "p95_inference_time_ms": metrics.get("p95_inference_time_ms"),
        "p99_inference_time_ms": metrics.get("p99_inference_time_ms"),
        "status": "experimental",
        "deployed_at": None,
        "deprecated_at": None,
        "accuracy_improvement": None,
        "latency_change_percent": None,
        "created_at": now.isoformat(),
        "created_by": os.environ.get("USER", "unknown"),
        "changelog": f"New model version {model_version}"
    }
    
    errors = client.insert_rows_json(table, [row])
    if errors:
        raise Exception(f"BigQuery model_versions insert errors: {errors}")
    return True


def save_experiment_metrics(run_id, metrics, training_time):
    """Save experiment metrics to BigQuery experiment_metrics table"""
    client = bigquery.Client(project=PROJECT_ID)
    table = f"{PROJECT_ID}.{BIGQUERY_DATASET}.experiment_metrics"
    now = datetime.now()
    metric_rows = []
    
    for metric_name, metric_value in [
        ("val_accuracy", metrics.get("val_accuracy")),
        ("val_precision", metrics.get("val_precision")),
        ("val_recall", metrics.get("val_recall")),
        ("val_f1", metrics.get("val_f1")),
        ("training_time_seconds", training_time),
        ("avg_confidence", metrics.get("avg_confidence")),
    ]:
        if metric_value is not None:
            metric_rows.append({
                "metric_id": f"{run_id}-{metric_name}",
                "run_id": run_id,
                "experiment_name": EXPERIMENT_NAME,
                "metric_type": metric_name.split("_")[0] if "_" in metric_name else metric_name,
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "metric_unit": "percentage" if "accuracy" in metric_name or "f1" in metric_name or "precision" in metric_name or "recall" in metric_name else "seconds" if "time" in metric_name else "float",
                "split": "val" if "val" in metric_name else "all",
                "category_id": None,
                "category_name": None,
                "timestamp": now.isoformat(),
                "notes": None
            })
    
    if metric_rows:
        errors = client.insert_rows_json(table, metric_rows)
        if errors:
            print(f"Warning: BigQuery experiment_metrics insert errors: {errors}")
        else:
            return len(metric_rows)
    return 0


def save_all_metadata(run_id, model_version, dataset_version, metrics, hyperparameters,
                     training_time, model_size_mb, model_gcs_path):
    """Save all metadata to BigQuery (wrapper function)"""
    save_training_run(run_id, model_version, dataset_version, metrics, hyperparameters,
                     training_time, model_size_mb, model_gcs_path)
    save_model_version(run_id, model_version, model_gcs_path, model_size_mb,
                      hyperparameters, metrics)
    metrics_count = save_experiment_metrics(run_id, metrics, training_time)
    return metrics_count

