"""
Model evaluation utilities
"""
import sys
import numpy as np
import pandas as pd
import json
import hashlib
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.category_names import load_category_names


def evaluate_model(classifier, df_train=None, df_val=None, confidence_threshold=0.5):
    """Evaluate model on train and validation sets"""
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    metrics = {}
    
    # Evaluation on train
    if df_train is not None:
        print("\nEvaluating on training data...")
        if classifier.X_train is not None and classifier.df_train is not None:
            print("   ✓ Reusing training embeddings")
            y_pred_encoded = classifier.classifier.predict(classifier.X_train)
            y_pred_train = classifier.label_encoder.inverse_transform(y_pred_encoded)
            y_pred_proba = classifier.classifier.predict_proba(classifier.X_train)
            conf_train = np.max(y_pred_proba, axis=1)
            df_train_eval = classifier.df_train
        else:
            y_pred_train, conf_train = classifier.predict_with_confidence(df_train)
            df_train_eval = df_train
        
        y_true_train = df_train_eval['category_id'].values
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_true_train, y_pred_train, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")
        
        metrics['train_accuracy'] = train_acc
        metrics['train_precision'] = train_prec
        metrics['train_recall'] = train_rec
        metrics['train_f1'] = train_f1
    
    # Evaluation on validation
    if df_val is not None:
        print("\nEvaluating on validation data...")
        
        # Cache embeddings
        cache_dir = Path(__file__).parent.parent.parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        cache_hash = hashlib.md5(str(id(df_val)).encode()).hexdigest()
        cache_path = cache_dir / f'val_embeddings_{cache_hash}.npy'
        
        X_val = classifier.prepare_features(df_val, show_progress=True, cache_path=cache_path)
        y_pred_encoded = classifier.classifier.predict(X_val)
        y_pred_val = classifier.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = classifier.classifier.predict_proba(X_val)
        conf_val = np.max(y_pred_proba, axis=1)
        y_true_val = df_val['category_id'].values
        
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            y_true_val, y_pred_val, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"   Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1: {val_f1:.4f}")
        
        metrics['val_accuracy'] = val_acc
        metrics['val_precision'] = val_prec
        metrics['val_recall'] = val_rec
        metrics['val_f1'] = val_f1
        metrics['avg_confidence'] = float(np.mean(conf_val))
        
        # Comparison train vs validation
        if df_train is not None and 'train_accuracy' in metrics:
            gap_acc = metrics['train_accuracy'] - val_acc
            print("\nComparison Train vs Validation:")
            print(f"   Accuracy gap: {gap_acc:.4f} ({gap_acc*100:.2f} points)")
            if gap_acc > 0.05:
                print(f"   ⚠️  Overfitting detected (gap > 5 points)")
            else:
                print(f"   ✓ No significant overfitting")
            metrics['gap'] = gap_acc
        
        # Detailed analysis
        analyze_categories(df_val, y_pred_val, conf_val, y_true_val, 
                          classifier.cat_to_path, confidence_threshold)
    
    return metrics


def analyze_categories(df, predictions, confidence_scores, y_true, cat_to_path, threshold=0.5):
    """Analyze categories: generate JSON files (certain, uncertain, confusion)"""
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    category_names = load_category_names()
    
    df_analysis = df.copy()
    df_analysis['predicted_category'] = predictions
    df_analysis['confidence'] = confidence_scores
    df_analysis['is_certain'] = confidence_scores >= threshold
    df_analysis['is_correct'] = predictions == y_true
    
    certain_stats = []
    uncertain_stats = []
    confusion_pairs = {}
    
    for cat_id in df['category_id'].unique():
        cat_data = df_analysis[df_analysis['category_id'] == cat_id]
        cat_name = category_names.get(cat_id, 'N/A')
        total_in_cat = len(cat_data)
        
        certain_data = cat_data[cat_data['is_certain']]
        uncertain_data = cat_data[~cat_data['is_certain']]
        
        if len(certain_data) > 0:
            confidences = certain_data['confidence'].values
            example_titles = []
            for idx in certain_data.head(3).index:
                row = certain_data.loc[idx]
                example_titles.append({
                    'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
                    'confidence': round(float(row['confidence']), 3),
                    'true_category': cat_id,
                    'true_path': row['category_path'],
                    'predicted_category': row['predicted_category'],
                    'predicted_path': cat_to_path.get(row['predicted_category'], 'N/A')
                })
            
            certain_stats.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'n_certain_products': len(certain_data),
                'avg_confidence': round(float(np.mean(confidences)), 3),
                'certainty_rate': round(len(certain_data) / total_in_cat, 3),
                'example_titles': example_titles
            })
        
        if len(uncertain_data) > 0:
            confidences = uncertain_data['confidence'].values
            example_titles = []
            for idx in uncertain_data.head(3).index:
                row = uncertain_data.loc[idx]
                example_titles.append({
                    'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
                    'confidence': round(float(row['confidence']), 3),
                    'true_category': cat_id,
                    'true_path': row['category_path'],
                    'predicted_category': row['predicted_category'],
                    'predicted_path': cat_to_path.get(row['predicted_category'], 'N/A')
                })
            
            uncertain_stats.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'n_uncertain_products': len(uncertain_data),
                'uncertainty_rate': round(len(uncertain_data) / total_in_cat, 3),
                'avg_confidence': round(float(np.mean(confidences)), 3),
                'example_titles': example_titles
            })
        
        # Confusion patterns
        errors = cat_data[cat_data['predicted_category'] != cat_id]
        for _, row in errors.iterrows():
            pred_cat = row['predicted_category']
            pair_key = (cat_id, pred_cat)
            
            if pair_key not in confusion_pairs:
                confusion_pairs[pair_key] = {'confidences': [], 'examples': []}
            
            confusion_pairs[pair_key]['confidences'].append(row['confidence'])
            if len(confusion_pairs[pair_key]['examples']) < 3:
                confusion_pairs[pair_key]['examples'].append({
                    'product_id': row['product_id'],
                    'title': str(row['title'])[:100] if pd.notna(row['title']) else ''
                })
    
    # Save Certain
    certain_stats.sort(key=lambda x: x['n_certain_products'], reverse=True)
    certain_df = df_analysis[df_analysis['is_certain']]
    output_path = Path(__file__).parent.parent.parent / 'results' / 'classification'
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / 'certain_categories_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_certain_products': len(certain_df),
                'total_categories_with_certain': len(certain_stats),
                'avg_confidence': round(float(np.mean(certain_df['confidence'])), 3),
                'threshold': threshold
            },
            'top_10_categories': certain_stats[:10]
        }, f, indent=2, ensure_ascii=False)
    
    # Save Uncertain
    uncertain_stats.sort(key=lambda x: x['n_uncertain_products'], reverse=True)
    uncertain_df = df_analysis[~df_analysis['is_certain']]
    
    with open(output_path / 'uncertain_categories_analysis.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_uncertain_products': len(uncertain_df),
                'total_categories_with_uncertain': len(uncertain_stats),
                'avg_confidence': round(float(np.mean(uncertain_df['confidence'])), 3),
                'threshold': threshold
            },
            'top_10_categories': uncertain_stats[:10]
        }, f, indent=2, ensure_ascii=False)
    
    # Save Confusion Patterns
    patterns = []
    errors_all = df_analysis[df_analysis['predicted_category'] != df_analysis['category_id']]
    for (true_cat, pred_cat), data in confusion_pairs.items():
        true_name = category_names.get(true_cat, 'N/A')
        pred_name = category_names.get(pred_cat, 'N/A')
        total_true_cat = len(df_analysis[df_analysis['category_id'] == true_cat])
        confusion_rate = len(data['confidences']) / total_true_cat if total_true_cat > 0 else 0
        
        patterns.append({
            'true_category_id': true_cat,
            'true_category_name': true_name,
            'predicted_category_id': pred_cat,
            'predicted_category_name': pred_name,
            'n_cases': len(data['confidences']),
            'confusion_rate': round(confusion_rate, 3),
            'avg_confidence': round(float(np.mean(data['confidences'])), 3),
            'examples': data['examples']
        })
    
    patterns.sort(key=lambda x: x['n_cases'], reverse=True)
    
    with open(output_path / 'confusion_patterns.json', 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_confusion_cases': len(errors_all),
                'unique_confusion_pairs': len(patterns)
            },
            'top_10_confusion_patterns': patterns[:10]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ {len(certain_stats)} certain categories, {len(uncertain_stats)} uncertain")
    print(f"   ✓ {len(patterns)} confusion patterns identified")
    print(f"   ✓ 3 JSON files generated")

