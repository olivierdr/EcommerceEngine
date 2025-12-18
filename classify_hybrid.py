"""
Classification Hybride
Flat classification avec validation hiérarchique : choisir parmi top-K cohérents
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
import json
import pickle
import warnings
warnings.filterwarnings('ignore')


class HybridClassifier:
    """Classifieur hybride : Réutilise le modèle flat + validation hiérarchique améliorée"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Charge le modèle flat pré-entraîné"""
        if model_path is None:
            model_path = Path(__file__).parent / 'flat_model.pkl'
        
        print("🔄 Chargement du modèle flat...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        embedding_model_name = model_data.get('embedding_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Charger les métadonnées depuis le modèle (production-ready)
        self.cat_to_path = model_data.get('cat_to_path', {})
        self.problematic_categories = set(model_data.get('problematic_categories', []))
        self.confidence_threshold = confidence_threshold
        
        if not self.cat_to_path:
            print("   ⚠️  cat_to_path non trouvé dans le modèle")
        if not self.problematic_categories:
            print("   ⚠️  problematic_categories non trouvé dans le modèle")
    
    def load_hierarchy(self, train_path=None):
        """Charge la hiérarchie depuis le modèle (déjà chargé) ou depuis train_path (fallback)"""
        if self.cat_to_path:
            print(f"   ✓ Hiérarchie chargée depuis le modèle ({len(self.cat_to_path)} catégories)")
            return self
        
        # Fallback : charger depuis train_path si pas dans le modèle
        if train_path:
            print("   ⚠️  Fallback : chargement depuis train_path...")
            df_train = pd.read_csv(train_path)
            self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
            print(f"   ✓ Hiérarchie chargée depuis train_path ({len(self.cat_to_path)} catégories)")
        else:
            print("   ⚠️  Aucune hiérarchie disponible")
        
        return self
    
    def get_parent(self, category_id, level_up=1):
        """Récupère la catégorie parente à N niveaux au-dessus"""
        if category_id not in self.cat_to_path:
            return None
        path = self.cat_to_path[category_id].split('/')
        return path[-(level_up + 1)] if len(path) > level_up else None
    
    def is_child_of(self, child_id, parent_id):
        """Vérifie si child_id est un enfant de parent_id"""
        if child_id not in self.cat_to_path or parent_id not in self.cat_to_path:
            return False
        child_path = self.cat_to_path[child_id].split('/')
        parent_path = self.cat_to_path[parent_id].split('/')
        return len(child_path) > len(parent_path) and child_path[:len(parent_path)] == parent_path
    
    def predict_with_confidence(self, df):
        """Prédit avec le modèle flat"""
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        X = self.embedding_model.encode(texts.tolist(), show_progress_bar=False, batch_size=128)
        y_pred_encoded = self.classifier.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = self.classifier.predict_proba(X)
        return y_pred, np.max(y_pred_proba, axis=1), y_pred_proba
    
    def validate_hierarchical(self, pred_cat, confidence, y_pred_proba_row, top_k=3):
        """Validation hiérarchique : choisir parmi top-K cohérents plutôt que remonter"""
        if confidence >= self.confidence_threshold:
            return pred_cat, "high_confidence"
        
        # Désactiver le fallback pour catégories problématiques
        if pred_cat in self.problematic_categories:
            return pred_cat, "low_confidence_no_fallback"
        
        # Vérifier cohérence hiérarchique parmi top-K
        top_k_indices = np.argsort(y_pred_proba_row)[-top_k:][::-1]
        top_k_categories = self.label_encoder.inverse_transform(top_k_indices)
        top_k_probas = y_pred_proba_row[top_k_indices]
        
        # Chercher un parent commun parmi les top-K
        if pred_cat in self.cat_to_path:
            pred_path = self.cat_to_path[pred_cat].split('/')
            
            for level in range(1, min(len(pred_path), 4)):
                parent_id = pred_path[-(level + 1)]
                coherent_cats = []
                coherent_probas = []
                
                # Identifier les top-K qui partagent ce parent
                for top_cat, top_prob in zip(top_k_categories, top_k_probas):
                    if top_cat in self.cat_to_path:
                        top_path = self.cat_to_path[top_cat].split('/')
                        if len(top_path) > level and top_path[-(level + 1)] == parent_id:
                            coherent_cats.append(top_cat)
                            coherent_probas.append(top_prob)
                
                # Si ≥2 top-K partagent un parent → choisir celui avec probabilité max
                if len(coherent_cats) >= 2:
                    best_idx = np.argmax(coherent_probas)
                    return coherent_cats[best_idx], "low_confidence_coherent_choice"
        
        return pred_cat, "low_confidence"
    
    def predict(self, df):
        """Prédit avec validation hiérarchique améliorée"""
        y_pred, confidence_scores, y_pred_proba = self.predict_with_confidence(df)
        validated_predictions, validation_flags = [], []
        
        for idx, (pred_cat, conf) in enumerate(zip(y_pred, confidence_scores)):
            validated_cat, flag = self.validate_hierarchical(pred_cat, conf, y_pred_proba[idx])
            validated_predictions.append(validated_cat)
            validation_flags.append(flag)
        
        return np.array(validated_predictions), confidence_scores, np.array(validation_flags)
    
    def calculate_hierarchical_accuracy(self, y_true, y_pred):
        """Calcule l'accuracy hiérarchique : correct si feuille exacte OU parent correct"""
        exact_correct = (y_true == y_pred).sum()
        hierarchical_correct = exact_correct
        
        # Pour les erreurs, vérifier si le parent est correct
        errors_mask = y_true != y_pred
        for idx in np.where(errors_mask)[0]:
            true_cat = y_true[idx]
            pred_cat = y_pred[idx]
            
            # Vérifier si pred_cat est un parent de true_cat
            if self.is_child_of(true_cat, pred_cat):
                hierarchical_correct += 1
        
        return exact_correct / len(y_true), hierarchical_correct / len(y_true)
    
    def identify_uncertain_products(self, df, predictions, confidence_scores, validation_flags):
        """Identifie les produits incertains pour validation humaine"""
        uncertain_mask = confidence_scores < self.confidence_threshold
        uncertain_data = []
        
        for idx in np.where(uncertain_mask)[0]:
            row = df.iloc[idx]
            uncertain_data.append({
                'product_id': row['product_id'],
                'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
                'true_category': row['category_id'],
                'true_path': row['category_path'],
                'predicted_category': predictions[idx],
                'predicted_path': self.cat_to_path.get(predictions[idx], 'N/A'),
                'confidence_score': round(float(confidence_scores[idx]), 3),
                'validation_flag': validation_flags[idx]
            })
        
        uncertain_data.sort(key=lambda x: x['confidence_score'])
        
        output_path = Path(__file__).parent / 'uncertain_products'
        output_path.mkdir(exist_ok=True)
        with open(output_path / 'uncertain_products.json', 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': self.confidence_threshold,
                'total_uncertain_products': len(uncertain_data),
                'products': uncertain_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(uncertain_data)} produits incertains sauvegardés")
        return uncertain_data
    
    def evaluate(self, test_path):
        """Évalue le modèle hybride avec accuracy exacte et hiérarchique"""
        print("\n" + "="*60)
        print("📊 ÉVALUATION - Classification Hybride")
        print("="*60)
        
        df_test = pd.read_csv(test_path)
        print(f"\n📊 {len(df_test):,} produits chargés")
        
        y_pred, confidence_scores, validation_flags = self.predict(df_test)
        y_true = df_test['category_id'].values
        
        # Accuracy exacte et hiérarchique
        acc_exact, acc_hierarchical = self.calculate_hierarchical_accuracy(y_true, y_pred)
        
        # Comparaison avec flat pur
        y_pred_flat, _, _ = self.predict_with_confidence(df_test)
        acc_flat_exact = accuracy_score(y_true, y_pred_flat)
        acc_flat_hierarchical, _ = self.calculate_hierarchical_accuracy(y_true, y_pred_flat)
        
        print(f"\n✅ Accuracy Exacte:")
        print(f"   Flat:    {acc_flat_exact:.4f} ({acc_flat_exact*100:.2f}%)")
        print(f"   Hybride: {acc_exact:.4f} ({acc_exact*100:.2f}%)")
        print(f"   Différence: {acc_exact - acc_flat_exact:+.4f} ({+(acc_exact - acc_flat_exact)*100:+.2f} points)")
        
        print(f"\n✅ Accuracy Hiérarchique (tolérante):")
        print(f"   Flat:    {acc_flat_hierarchical:.4f} ({acc_flat_hierarchical*100:.2f}%)")
        print(f"   Hybride: {acc_hierarchical:.4f} ({acc_hierarchical*100:.2f}%)")
        print(f"   Différence: {acc_hierarchical - acc_flat_hierarchical:+.4f} ({+(acc_hierarchical - acc_flat_hierarchical)*100:+.2f} points)")
        
        # Statistiques de validation
        high_conf = (validation_flags == 'high_confidence').sum()
        low_conf = (validation_flags == 'low_confidence').sum()
        coherent = (validation_flags == 'low_confidence_coherent_choice').sum()
        no_fallback = (validation_flags == 'low_confidence_no_fallback').sum()
        
        print(f"\n📈 Répartition des prédictions:")
        print(f"   Haute confiance: {high_conf} ({high_conf/len(y_pred)*100:.1f}%)")
        print(f"   Choix cohérent: {coherent} ({coherent/len(y_pred)*100:.1f}%)")
        print(f"   Pas de fallback (problématique): {no_fallback} ({no_fallback/len(y_pred)*100:.1f}%)")
        print(f"   Faible confiance: {low_conf} ({low_conf/len(y_pred)*100:.1f}%)")
        
        self.identify_uncertain_products(df_test, y_pred, confidence_scores, validation_flags)
        
        return {
            'accuracy_exact': acc_exact,
            'accuracy_hierarchical': acc_hierarchical,
            'accuracy_flat_exact': acc_flat_exact,
            'accuracy_flat_hierarchical': acc_flat_hierarchical
        }


def main():
    """Point d'entrée principal"""
    base_path = Path(__file__).parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    model_path = base_path / 'flat_model.pkl'
    
    if not model_path.exists():
        print(f"❌ Modèle flat non trouvé. Exécutez d'abord: python3 classify_flat.py")
        return
    
    print("\n" + "="*60)
    print("🚀 CLASSIFICATION HYBRIDE")
    print("="*60)
    
    classifier = HybridClassifier(model_path=model_path, confidence_threshold=0.5)
    classifier.load_hierarchy(train_path)  # Fallback si cat_to_path manquant dans le modèle
    results = classifier.evaluate(test_path)
    
    print("\n" + "="*60)
    print("✓ Classification Hybride terminée")
    print("="*60)


if __name__ == '__main__':
    main()
