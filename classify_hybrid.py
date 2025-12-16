"""
Classification Hybride
Flat classification avec validation hiérarchique et gestion des produits incertains

L'approche hybride réutilise le modèle flat entraîné et ajoute :
1. Identification des produits incertains pour validation humaine
2. Validation hiérarchique pour produits à faible confiance
3. Métriques de confiance pour monitoring
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
    """Classifieur hybride : Réutilise le modèle flat + validation hiérarchique"""
    
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """Charge le modèle flat pré-entraîné"""
        if model_path is None:
            model_path = Path(__file__).parent / 'flat_model.pkl'
        
        print("🔄 Chargement du modèle flat pré-entraîné...")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.brand_encoder = model_data.get('brand_encoder')
        self.color_encoder = model_data.get('color_encoder')
        self.brand_onehot = model_data.get('brand_onehot')
        self.color_onehot = model_data.get('color_onehot')
        
        # Charger le modèle d'embeddings
        embedding_model_name = model_data.get('embedding_model_name', 'paraphrase-multilingual-MiniLM-L12-v2')
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.cat_to_path = {}
        self.confidence_threshold = confidence_threshold
        print("   ✓ Modèle chargé")
    
    def prepare_features(self, df):
        """Prépare les features : réutilise la même logique que flat"""
        # Features textuelles
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        text_embeddings = self.embedding_model.encode(texts.tolist(), show_progress_bar=False)
        
        # Features catégorielles (brand et color)
        brands = df['brand'].fillna('unknown').astype(str)
        colors = df['color'].fillna('unknown').astype(str)
        
        brands_known = brands.map(lambda x: x if x in self.brand_encoder.classes_ else 'unknown')
        colors_known = colors.map(lambda x: x if x in self.color_encoder.classes_ else 'unknown')
        brand_encoded = self.brand_encoder.transform(brands_known)
        color_encoded = self.color_encoder.transform(colors_known)
        
        brand_onehot = self.brand_onehot.transform(brand_encoded.reshape(-1, 1))
        color_onehot = self.color_onehot.transform(color_encoded.reshape(-1, 1))
        
        # Concaténer toutes les features
        features = np.hstack([text_embeddings, brand_onehot, color_onehot])
        return features
    
    def load_hierarchy(self, train_path):
        """Charge la hiérarchie depuis les données d'entraînement"""
        df_train = pd.read_csv(train_path)
        self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
        return self
    
    def predict_with_confidence(self, df):
        """Prédit avec scores de confiance"""
        X = self.prepare_features(df)
        y_pred_encoded = self.classifier.predict(X)
        y_pred_proba = self.classifier.predict_proba(X)
        
        # Confiance = probabilité maximale
        confidence_scores = np.max(y_pred_proba, axis=1)
        
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        return y_pred, confidence_scores, y_pred_proba
    
    def get_parent_category(self, category_id, level_up=1):
        """Récupère la catégorie parente à N niveaux au-dessus"""
        if category_id not in self.cat_to_path:
            return None
        
        path = self.cat_to_path[category_id].split('/')
        if len(path) > level_up:
            return path[-(level_up + 1)]
        return None
    
    def validate_hierarchical(self, category_id, confidence, y_pred_proba_row, top_k=3):
        """Validation hiérarchique améliorée : utilise hiérarchie pour améliorer les prédictions"""
        if confidence >= self.confidence_threshold:
            return category_id, confidence, "high_confidence"
        
        # Confiance faible : vérifier si les top-K prédictions sont dans la même branche hiérarchique
        top_k_indices = np.argsort(y_pred_proba_row)[-top_k:][::-1]
        top_k_categories = self.label_encoder.inverse_transform(top_k_indices)
        top_k_probas = y_pred_proba_row[top_k_indices]
        
        # Vérifier si plusieurs top prédictions partagent un parent commun
        if category_id in self.cat_to_path:
            pred_path = self.cat_to_path[category_id].split('/')
            
            # Chercher un parent commun parmi les top-K
            for level in range(1, min(len(pred_path), 4)):  # Vérifier jusqu'au niveau 4
                parent_id = pred_path[-(level + 1)]
                # Compter combien de top-K ont ce parent
                common_parent_count = 0
                for top_cat in top_k_categories:
                    if top_cat in self.cat_to_path:
                        top_path = self.cat_to_path[top_cat].split('/')
                        if len(top_path) > level and top_path[-(level + 1)] == parent_id:
                            common_parent_count += 1
                
                # Si au moins 2 top-K partagent un parent, on peut être plus confiant
                if common_parent_count >= 2:
                    return category_id, confidence, "low_confidence_but_coherent"
        
        return category_id, confidence, "low_confidence"
    
    def predict(self, df):
        """Prédit avec validation hiérarchique améliorée"""
        y_pred, confidence_scores, y_pred_proba = self.predict_with_confidence(df)
        
        # Validation hiérarchique pour produits à faible confiance
        validated_predictions = []
        validation_flags = []
        
        for idx, (cat_id, conf) in enumerate(zip(y_pred, confidence_scores)):
            validated_cat, validated_conf, flag = self.validate_hierarchical(
                cat_id, conf, y_pred_proba[idx]
            )
            validated_predictions.append(validated_cat)
            validation_flags.append(flag)
        
        return np.array(validated_predictions), confidence_scores, np.array(validation_flags)
    
    def identify_uncertain_products(self, df, predictions, confidence_scores, validation_flags, output_dir='uncertain_products'):
        """Identifie et sauvegarde les produits avec faible confiance en JSON"""
        print(f"\n🔍 Identification des produits incertains (confiance < {self.confidence_threshold})...")
        
        # Créer le dossier de sortie
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        # Identifier les produits incertains
        uncertain_mask = confidence_scores < self.confidence_threshold
        uncertain_data = []
        
        for idx in np.where(uncertain_mask)[0]:
            row = df.iloc[idx]
            pred_cat = predictions[idx]
            true_cat = row['category_id']
            conf = confidence_scores[idx]
            flag = validation_flags[idx]
            
            # Récupérer le path prédit et réel
            predicted_path = self.cat_to_path.get(pred_cat, 'N/A')
            true_path = row['category_path']
            
            # Title court (max 100 caractères)
            title = str(row['title'])[:100] if pd.notna(row['title']) else ''
            
            product_info = {
                'product_id': row['product_id'],
                'title': title,
                'true_category': true_cat,
                'true_path': true_path,
                'predicted_category': pred_cat,
                'predicted_path': predicted_path,
                'confidence_score': float(conf),
                'validation_flag': flag
            }
            uncertain_data.append(product_info)
        
        # Trier par confiance (croissant)
        uncertain_data.sort(key=lambda x: x['confidence_score'])
        
        # Sauvegarder en JSON
        output_file = output_path / 'uncertain_products.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': self.confidence_threshold,
                'total_uncertain_products': len(uncertain_data),
                'products': uncertain_data
            }, f, indent=2, ensure_ascii=False)
        
        n_uncertain = len(uncertain_data)
        avg_conf = np.mean([p['confidence_score'] for p in uncertain_data]) if uncertain_data else 0
        min_conf = min([p['confidence_score'] for p in uncertain_data]) if uncertain_data else 0
        max_conf = max([p['confidence_score'] for p in uncertain_data]) if uncertain_data else 0
        
        print(f"   ✓ {n_uncertain} produits incertains identifiés")
        print(f"   ✓ Sauvegardés dans: {output_file}")
        print(f"   📊 Confiance moyenne: {avg_conf:.3f}")
        print(f"   📊 Confiance min: {min_conf:.3f}")
        print(f"   📊 Confiance max: {max_conf:.3f}")
        
        return uncertain_data
    
    def evaluate(self, test_path):
        """Évalue le modèle avec métriques améliorées"""
        print("\n" + "="*60)
        print("📊 ÉVALUATION - Classification Hybride")
        print("="*60)
        
        # Charger les données de test
        print("\n📊 Chargement des données de test...")
        df_test = pd.read_csv(test_path)
        print(f"   ✓ {len(df_test):,} produits chargés")
        
        # Prédictions avec validation
        print("\n🔮 Prédictions avec validation hiérarchique...")
        y_pred, confidence_scores, validation_flags = self.predict(df_test)
        y_true = df_test['category_id'].values
        
        # Accuracy globale
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✅ Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Accuracy par niveau de confiance
        high_conf_mask = validation_flags == 'high_confidence'
        low_conf_mask = validation_flags == 'low_confidence'
        
        if high_conf_mask.sum() > 0:
            acc_high = accuracy_score(y_true[high_conf_mask], y_pred[high_conf_mask])
            print(f"\n📊 Accuracy produits haute confiance (≥{self.confidence_threshold}):")
            print(f"   {acc_high:.4f} ({acc_high*100:.2f}%) - {high_conf_mask.sum()} produits")
        
        if low_conf_mask.sum() > 0:
            acc_low = accuracy_score(y_true[low_conf_mask], y_pred[low_conf_mask])
            print(f"\n⚠️  Accuracy produits faible confiance (<{self.confidence_threshold}):")
            print(f"   {acc_low:.4f} ({acc_low*100:.2f}%) - {low_conf_mask.sum()} produits")
        
        # Statistiques de confiance
        print(f"\n📈 Statistiques de confiance:")
        print(f"   Confiance moyenne: {confidence_scores.mean():.4f}")
        print(f"   Confiance médiane: {np.median(confidence_scores):.4f}")
        print(f"   Confiance min: {confidence_scores.min():.4f}")
        print(f"   Confiance max: {confidence_scores.max():.4f}")
        
        # Identifier les produits incertains
        uncertain_products = self.identify_uncertain_products(df_test, y_pred, confidence_scores, validation_flags)
        
        # Comparaison avec flat pur (sans validation)
        y_pred_flat, _, _ = self.predict_with_confidence(df_test)
        accuracy_flat = accuracy_score(y_true, y_pred_flat)
        
        print(f"\n📊 Comparaison avec flat pur:")
        print(f"   Flat pur: {accuracy_flat:.4f} ({accuracy_flat*100:.2f}%)")
        print(f"   Hybride:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        improvement = accuracy - accuracy_flat
        if improvement > 0:
            print(f"   ➕ Amélioration: +{improvement:.4f} (+{improvement*100:.2f} points)")
        else:
            print(f"   ➖ Différence: {improvement:.4f} ({improvement*100:.2f} points)")
        
        return {
            'accuracy': accuracy,
            'accuracy_flat': accuracy_flat,
            'y_true': y_true,
            'y_pred': y_pred,
            'confidence_scores': confidence_scores,
            'validation_flags': validation_flags,
            'uncertain_products': uncertain_products
        }


def main():
    """Point d'entrée principal"""
    base_path = Path(__file__).parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    model_path = base_path / 'flat_model.pkl'
    
    # Vérifier que le modèle existe
    if not model_path.exists():
        print(f"❌ Modèle flat non trouvé: {model_path}")
        print("   Veuillez d'abord exécuter: python3 classify_flat.py")
        return
    
    if not test_path.exists():
        print(f"❌ Fichier non trouvé: {test_path}")
        return
    
    # Charger le modèle flat pré-entraîné
    print("\n" + "="*60)
    print("🚀 CLASSIFICATION HYBRIDE")
    print("="*60)
    classifier = HybridClassifier(model_path=model_path, confidence_threshold=0.5)
    
    # Charger la hiérarchie
    classifier.load_hierarchy(train_path)
    
    # Évaluer sur le test set
    results = classifier.evaluate(test_path)
    
    print("\n" + "="*60)
    print("✓ Classification Hybride terminée")
    print("="*60)
    print(f"\n💡 Les produits incertains sont disponibles pour validation humaine")
    print(f"   dans le dossier: uncertain_products/")
    
    return classifier, results


if __name__ == '__main__':
    classifier, results = main()
