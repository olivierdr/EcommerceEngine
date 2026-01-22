"""
Évaluation du classifieur et analyses détaillées
"""

import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from train import FlatClassifier, load_category_names
import warnings
warnings.filterwarnings('ignore')


def evaluate(classifier, train_path, val_path, confidence_threshold=0.5):
    """Évalue le modèle sur train et validation"""
    print("\n" + "="*60)
    print("ÉVALUATION")
    print("="*60)
    
    # Évaluation sur train
    print("\nÉvaluation sur données d'entraînement...")
    if classifier.X_train is not None and classifier.df_train is not None:
        print("   ✓ Réutilisation des embeddings d'entraînement")
        y_pred_encoded = classifier.classifier.predict(classifier.X_train)
        y_pred_train = classifier.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = classifier.classifier.predict_proba(classifier.X_train)
        conf_train = np.max(y_pred_proba, axis=1)
        df_train = classifier.df_train
    else:
        df_train = pd.read_csv(train_path)
        y_pred_train, conf_train = classifier.predict_with_confidence(df_train)
    
    y_true_train = df_train['category_id'].values
    train_acc = accuracy_score(y_true_train, y_pred_train)
    train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
        y_true_train, y_pred_train, average='weighted', zero_division=0
    )
    
    print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"   Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")
    
    # Évaluation sur validation
    print("\nÉvaluation sur données de validation...")
    df_test = pd.read_csv(val_path)
    
    # Cache des embeddings validation
    cache_dir = Path(__file__).parent.parent / '.cache'
    cache_dir.mkdir(exist_ok=True)
    cache_hash = hashlib.md5(str(val_path).encode()).hexdigest()
    cache_path = cache_dir / f'val_embeddings_{cache_hash}.npy'
    
    X_test = classifier.prepare_features(df_test, show_progress=True, cache_path=cache_path)
    y_pred_encoded = classifier.classifier.predict(X_test)
    y_pred_test = classifier.label_encoder.inverse_transform(y_pred_encoded)
    y_pred_proba = classifier.classifier.predict_proba(X_test)
    conf_test = np.max(y_pred_proba, axis=1)
    y_true_test = df_test['category_id'].values
    
    test_acc = accuracy_score(y_true_test, y_pred_test)
    test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
        y_true_test, y_pred_test, average='weighted', zero_division=0
    )
    
    print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
    
    # Comparaison train vs validation
    print("\nComparaison Train vs Validation:")
    gap_acc = train_acc - test_acc
    print(f"   Écart Accuracy: {gap_acc:.4f} ({gap_acc*100:.2f} points)")
    
    if gap_acc > 0.05:
        print(f"   ⚠️  Sur-apprentissage détecté (écart > 5 points)")
    else:
        print(f"   ✓ Pas de sur-apprentissage significatif")
    
    # Analyses détaillées
    analyze_categories(df_test, y_pred_test, conf_test, y_true_test, 
                      classifier.cat_to_path, confidence_threshold)
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'gap': gap_acc,
        'y_true': y_true_test,
        'y_pred': y_pred_test
    }


def analyze_categories(df, predictions, confidence_scores, y_true, cat_to_path, threshold=0.5):
    """Analyse : génère les 3 JSON (certain, uncertain, confusion)"""
    print("\n" + "="*60)
    print("ANALYSES DÉTAILLÉES")
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
    
    # Sauvegarder Certain
    certain_stats.sort(key=lambda x: x['n_certain_products'], reverse=True)
    certain_df = df_analysis[df_analysis['is_certain']]
    output_path = Path(__file__).parent.parent / 'results' / 'classification'
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
    
    # Sauvegarder Uncertain
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
    
    # Sauvegarder Confusion Patterns
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
    
    print(f"   ✓ {len(certain_stats)} catégories certaines, {len(uncertain_stats)} incertaines")
    print(f"   ✓ {len(patterns)} patterns de confusion identifiés")
    print(f"   ✓ 3 fichiers JSON générés")


def main():
    print("\n" + "="*60)
    print("ÉVALUATION DU CLASSIFIEUR")
    print("="*60)
    
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'data' / 'trainset.csv'
    val_path = base_path / 'data' / 'valset.csv'
    model_path = base_path / 'results' / 'classification' / 'flat_model.pkl'
    
    if not train_path.exists() or not val_path.exists():
        print(f"Fichiers de données non trouvés")
        return None
    
    # Charger ou entraîner le modèle
    if model_path.exists():
        print("\nChargement du modèle existant...")
        classifier = FlatClassifier.load(model_path)
        # Charger les données d'entraînement pour réutiliser les embeddings
        classifier.df_train = pd.read_csv(train_path)
        classifier.X_train = classifier.prepare_features(classifier.df_train, show_progress=True)
    else:
        print("\nModèle non trouvé, entraînement...")
        from train import main as train_main
        classifier = train_main()
    
    # Évaluer
    results = evaluate(classifier, train_path, val_path, confidence_threshold=0.5)
    
    # Sauvegarder le modèle mis à jour
    classifier.save(model_path)
    
    print("\n" + "="*60)
    print("✓ Évaluation terminée")
    print("="*60)
    
    return results


if __name__ == '__main__':
    results = main()

