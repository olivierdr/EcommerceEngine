"""
Analyse comparative détaillée : Flat vs Hybride
Génère un rapport complet avec métriques, erreurs et recommandations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from classify_flat import FlatClassifier
from classify_hybrid import HybridClassifier
import json
import time
import warnings
warnings.filterwarnings('ignore')


def analyze_errors(y_true, y_pred, df_test, top_n=10):
    """Analyse les erreurs : top catégories confondues et exemples"""
    errors = y_true != y_pred
    error_mask = np.where(errors)[0]
    
    # Top catégories avec le plus d'erreurs
    error_categories = pd.Series(y_true[error_mask]).value_counts().head(top_n)
    
    # Paires les plus confondues (true -> predicted)
    confusion_pairs = {}
    for idx in error_mask:
        true_cat = y_true[idx]
        pred_cat = y_pred[idx]
        pair_key = f"{true_cat} -> {pred_cat}"
        confusion_pairs[pair_key] = confusion_pairs.get(pair_key, 0) + 1
    
    top_confusions = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Exemples d'erreurs
    error_examples = []
    for idx in error_mask[:5]:
        row = df_test.iloc[idx]
        error_examples.append({
            'product_id': row['product_id'],
            'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
            'true_category': y_true[idx],
            'true_path': row['category_path'],
            'predicted_category': y_pred[idx]
        })
    
    return {
        'top_error_categories': error_categories.to_dict(),
        'top_confusions': [{'pair': k, 'count': v} for k, v in top_confusions],
        'error_examples': error_examples,
        'total_errors': int(errors.sum()),
        'error_rate': float(errors.sum() / len(y_true))
    }


def compare_approaches():
    """Compare les deux approches et génère un rapport complet"""
    print("="*60)
    print("📊 ANALYSE COMPARATIVE : Flat vs Hybride")
    print("="*60)
    
    base_path = Path(__file__).parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    # Entraîner et évaluer Flat
    print("\n1️⃣  Évaluation de l'approche Flat...")
    start_time = time.time()
    flat_classifier = FlatClassifier()
    flat_classifier.train(train_path)
    flat_results = flat_classifier.evaluate(test_path)
    flat_time = time.time() - start_time
    
    # Entraîner et évaluer Hybride
    print("\n2️⃣  Évaluation de l'approche Hybride...")
    start_time = time.time()
    hybrid_classifier = HybridClassifier()
    hybrid_classifier.train(train_path)
    hybrid_results = hybrid_classifier.evaluate(test_path)
    hybrid_time = time.time() - start_time
    
    # Charger les données de test pour analyse
    df_test = pd.read_csv(test_path)
    y_true = df_test['category_id'].values
    
    # Métriques détaillées
    print("\n" + "="*60)
    print("📈 MÉTRIQUES DÉTAILLÉES")
    print("="*60)
    
    # Précision/rappel/F1 par catégorie (top 5 seulement pour simplifier)
    top_categories = pd.Series(y_true).value_counts().head(5).index
    category_metrics = []
    for cat in top_categories:
        flat_tp = ((y_true == cat) & (flat_results['y_pred'] == cat)).sum()
        hybrid_tp = ((y_true == cat) & (hybrid_results['y_pred'] == cat)).sum()
        cat_count = (y_true == cat).sum()
        flat_pred_count = (flat_results['y_pred'] == cat).sum()
        hybrid_pred_count = (hybrid_results['y_pred'] == cat).sum()
        
        category_metrics.append({
            'category': cat,
            'flat': {'precision': float(flat_tp/flat_pred_count) if flat_pred_count > 0 else 0,
                    'recall': float(flat_tp/cat_count) if cat_count > 0 else 0},
            'hybrid': {'precision': float(hybrid_tp/hybrid_pred_count) if hybrid_pred_count > 0 else 0,
                      'recall': float(hybrid_tp/cat_count) if cat_count > 0 else 0}
        })
    
    # Analyse des erreurs
    print("\n🔍 Analyse des erreurs...")
    flat_errors = analyze_errors(y_true, flat_results['y_pred'], df_test)
    hybrid_errors = analyze_errors(y_true, hybrid_results['y_pred'], df_test)
    
    # Compilation du rapport
    report = {
        'comparison': {
            'flat': {
                'accuracy': float(flat_results['accuracy']),
                'training_time': round(flat_time, 2),
                'total_errors': flat_errors['total_errors'],
                'error_rate': flat_errors['error_rate']
            },
            'hybrid': {
                'accuracy': float(hybrid_results['accuracy']),
                'training_time': round(hybrid_time, 2),
                'total_errors': hybrid_errors['total_errors'],
                'error_rate': hybrid_errors['error_rate'],
                'uncertain_products': hybrid_results.get('uncertain_products', [])
            }
        },
        'top_categories_metrics': category_metrics,
        'error_analysis': {
            'flat': {
                'top_error_categories': flat_errors['top_error_categories'],
                'top_confusions': flat_errors['top_confusions'],
                'error_examples': flat_errors['error_examples']
            },
            'hybrid': {
                'top_error_categories': hybrid_errors['top_error_categories'],
                'top_confusions': hybrid_errors['top_confusions'],
                'error_examples': hybrid_errors['error_examples']
            }
        },
        'recommendations': {
            'best_approach': 'hybrid' if hybrid_results['accuracy'] >= flat_results['accuracy'] else 'flat',
            'reasoning': 'Hybrid offre validation humaine ciblée même si accuracy identique',
            'use_cases': {
                'flat': 'Production simple, performance maximale requise',
                'hybrid': 'Production avec validation humaine, monitoring qualité'
            }
        }
    }
    
    # Sauvegarder le rapport JSON
    report_path = base_path / 'comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("📋 RÉSUMÉ COMPARATIF")
    print("="*60)
    print(f"\n✅ Accuracy:")
    print(f"   Flat:   {flat_results['accuracy']:.4f} ({flat_results['accuracy']*100:.2f}%)")
    print(f"   Hybride: {hybrid_results['accuracy']:.4f} ({hybrid_results['accuracy']*100:.2f}%)")
    
    print(f"\n⏱️  Temps d'entraînement:")
    print(f"   Flat:   {flat_time:.2f}s")
    print(f"   Hybride: {hybrid_time:.2f}s")
    
    print(f"\n❌ Erreurs:")
    print(f"   Flat:   {flat_errors['total_errors']} ({flat_errors['error_rate']*100:.2f}%)")
    print(f"   Hybride: {hybrid_errors['total_errors']} ({hybrid_errors['error_rate']*100:.2f}%)")
    
    print(f"\n💡 Recommandation: {report['recommendations']['best_approach'].upper()}")
    print(f"   {report['recommendations']['reasoning']}")
    
    print(f"\n📄 Rapport complet sauvegardé: {report_path}")
    
    return report


if __name__ == '__main__':
    report = compare_approaches()

