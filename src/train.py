"""
E-commerce product classifier training - Main script
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import PROJECT_ID, REGION, EXPERIMENT_NAME, GCS_BUCKET
from src.data.loader import load_dataset
from src.training.trainer import train_model
from src.training.evaluator import evaluate_model
from src.tracking.vertex_ai import init_vertex_ai, start_run, log_params, log_metrics, end_run
from src.tracking.gcs import upload_model, upload_metadata
from src.tracking.bigquery import save_all_metadata
from src.utils.category_names import generate_category_names


def main(version="v1.0.0", dataset_version="v1.0", local_only=False):
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING")
    print("="*60)
    print(f"Model version: {version}")
    print(f"Dataset version: {dataset_version}")
    print(f"Local only: {local_only}")
    print("="*60)
    
    base_path = Path(__file__).parent.parent
    
    # Initialize Vertex AI (skip if local_only)
    run_id = None
    hyperparameters = {
        "max_iter": 1000,
        "random_state": 42,
        "solver": "lbfgs",
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
    }
    
    if not local_only:
        print("\nInitializing Vertex AI...")
        try:
            init_vertex_ai()
            run_id = start_run()
            print(f"✓ Run created: {run_id}")
            
            print(f"\nLogging hyperparameters...")
            log_params(hyperparameters)
            print("✓ Hyperparameters logged")
        except Exception as e:
            print(f"Warning: Could not initialize Vertex AI: {e}")
            print("Continuing without Vertex AI tracking...")
            run_id = None
    else:
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print("\nRunning in local-only mode (no Vertex AI, GCS, or BigQuery)")
    
    # Load datasets
    datasets = load_dataset(dataset_version, local_only=local_only, base_path=base_path)
    
    if 'train' not in datasets:
        print(f"Error: Training dataset not found")
        if run_id and not local_only:
            try:
                end_run()
            except:
                pass
        return None
    
    df_train = datasets['train']
    df_val = datasets.get('val', None)
    df_test = datasets.get('test', None)
    
    # Train model
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    classifier, training_time = train_model(df_train, hyperparameters)
    
    # Generate category names
    generate_category_names(classifier.df_train)
    
    # Evaluate model
    metrics = {}
    if df_val is not None:
        metrics = evaluate_model(classifier, df_train=df_train, df_val=df_val)
    else:
        print("\nNo validation set available, skipping evaluation")
        metrics = {
            "train_samples": len(df_train),
            "val_samples": 0,
            "test_samples": len(df_test) if df_test is not None else 0
        }
    
    # Add sample counts to metrics
    metrics['train_samples'] = len(df_train)
    metrics['val_samples'] = len(df_val) if df_val is not None else 0
    metrics['test_samples'] = len(df_test) if df_test is not None else 0
    
    # Save locally
    model_path = base_path / 'results' / 'classification' / 'flat_model.pkl'
    classifier.save(model_path)
    print(f"✓ Model saved locally: {model_path}")
    
    # Upload to GCS and save to BigQuery
    model_gcs_path = None
    if not local_only:
        print("\n" + "="*60)
        print("UPLOADING TO GCS AND BIGQUERY")
        print("="*60)
        
        try:
            # Calculate model size
            model_size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Upload model to GCS
            print(f"\nUploading model to GCS...")
            model_gcs_path = upload_model(model_path, version)
            print(f"✓ Model uploaded: {model_gcs_path}")
            
            # Upload metadata
            metadata = {
                "version": version,
                "dataset_version": dataset_version,
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "hyperparameters": hyperparameters,
                "metrics": metrics,
                "training_time_seconds": training_time
            }
            upload_metadata(metadata, version, base_path=base_path / 'results' / 'classification')
            print(f"✓ Metadata uploaded")
            
            # Log metrics to Vertex AI
            if run_id:
                try:
                    print("\nLogging metrics to Vertex AI...")
                    log_metrics({
                        "val_accuracy": metrics.get("val_accuracy", 0),
                        "val_precision": metrics.get("val_precision", 0),
                        "val_recall": metrics.get("val_recall", 0),
                        "val_f1": metrics.get("val_f1", 0),
                        "training_time_seconds": training_time,
                        "avg_confidence": metrics.get("avg_confidence", 0)
                    })
                    print("✓ Metrics logged")
                except Exception as e:
                    print(f"Warning: Could not log metrics: {e}")
            
            # Save to BigQuery
            print(f"\nSaving to BigQuery...")
            metrics_count = save_all_metadata(
                run_id=run_id,
                model_version=version,
                dataset_version=dataset_version,
                metrics=metrics,
                hyperparameters=hyperparameters,
                training_time=training_time,
                model_size_mb=model_size_mb,
                model_gcs_path=model_gcs_path
            )
            print(f"✓ Saved to BigQuery ({metrics_count} metrics)")
            
        except Exception as e:
            print(f"Warning: Could not upload to GCS/BigQuery: {e}")
            import traceback
            traceback.print_exc()
    
    # End run
    if run_id and not local_only:
        try:
            end_run()
            print(f"\n✓ Run completed: {run_id}")
        except Exception as e:
            print(f"Warning: Could not end run: {e}")
    
    print("\n" + "="*60)
    print("Training completed")
    print("="*60)
    
    return classifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train e-commerce product classifier')
    parser.add_argument('--version', type=str, default='v1.0.0', 
                       help='Model version (e.g., v1.0.0, v1.1.0)')
    parser.add_argument('--dataset-version', type=str, default='v1.0',
                       help='Dataset version (e.g., v1.0, v1.1)')
    parser.add_argument('--local-only', action='store_true',
                       help='Run in local-only mode (no GCS, BigQuery, or Vertex AI)')
    
    args = parser.parse_args()
    
    classifier = main(
        version=args.version,
        dataset_version=args.dataset_version,
        local_only=args.local_only
    )
