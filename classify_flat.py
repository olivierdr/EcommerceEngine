"""
Classification Flat (Baseline)
Prédiction directe de la catégorie feuille parmi les 100 classes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class FlatClassifier:
    """Classifieur flat pour prédire directement les catégories feuilles"""
    
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialise le modèle d'embeddings"""
        print("🔄 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.classifier = None
        self.label_encoder = None
        
    def prepare_features(self, df):
        """Prépare les features textuelles uniquement (title + description)"""
        print("📝 Préparation des features...")
        # Features textuelles uniquement
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        text_embeddings = self.embedding_model.encode(texts.tolist(), show_progress_bar=False)
        return text_embeddings
    
    def train(self, train_path):
        """Entraîne le modèle sur le train set"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT - Classification Flat")
        print("="*60)
        
        # Charger les données
        print("\n📊 Chargement des données d'entraînement...")
        df_train = pd.read_csv(train_path)
        print(f"   ✓ {len(df_train):,} produits chargés")
        
        # Préparer les features
        X_train = self.prepare_features(df_train)
        y_train = df_train['category_id'].values
        
        # Encoder les labels
        from sklearn.preprocessing import LabelEncoder
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\n📈 Statistiques:")
        print(f"   Nombre de catégories: {len(self.label_encoder.classes_)}")
        print(f"   Dimension des embeddings: {X_train.shape[1]}")
        
        # Entraîner le classifieur
        print("\n🎯 Entraînement du classifieur (Logistic Regression)...")
        self.classifier = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.classifier.fit(X_train, y_train_encoded)
        print("   ✓ Modèle entraîné")
        
        return self
    
    def predict(self, df):
        """Prédit les catégories pour un dataframe"""
        X = self.prepare_features(df)
        y_pred_encoded = self.classifier.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        return y_pred
    
    def predict_proba(self, df):
        """Retourne les probabilités de prédiction"""
        X = self.prepare_features(df)
        return self.classifier.predict_proba(X)
    
    def predict_with_confidence(self, df):
        """Prédit avec scores de confiance"""
        y_pred = self.predict(df)
        y_pred_proba = self.predict_proba(df)
        confidence_scores = np.max(y_pred_proba, axis=1)
        return y_pred, confidence_scores
    
    def evaluate(self, train_path, test_path, confidence_threshold=0.5):
        """Évalue le modèle sur train et test pour détecter le sur-apprentissage"""
        print("\n" + "="*60)
        print("📊 ÉVALUATION - Classification Flat")
        print("="*60)
        
        # Évaluation sur train
        print("\n📊 Évaluation sur données d'entraînement...")
        df_train = pd.read_csv(train_path)
        y_pred_train, conf_train = self.predict_with_confidence(df_train)
        y_true_train = df_train['category_id'].values
        
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_true_train, y_pred_train, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")
        
        # Évaluation sur test
        print("\n📊 Évaluation sur données de test...")
        df_test = pd.read_csv(test_path)
        y_pred_test, conf_test = self.predict_with_confidence(df_test)
        y_true_test = df_test['category_id'].values
        
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_true_test, y_pred_test, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
        
        # Comparaison train vs test
        print("\n📈 Comparaison Train vs Test:")
        gap_acc = train_acc - test_acc
        gap_prec = train_prec - test_prec
        gap_rec = train_rec - test_rec
        gap_f1 = train_f1 - test_f1
        
        print(f"   Écart Accuracy: {gap_acc:.4f} ({gap_acc*100:.2f} points)")
        print(f"   Écart Precision: {gap_prec:.4f} | Recall: {gap_rec:.4f} | F1: {gap_f1:.4f}")
        
        if gap_acc > 0.05:
            print(f"   ⚠️  Sur-apprentissage détecté (écart > 5 points)")
        else:
            print(f"   ✅ Pas de sur-apprentissage significatif")
        
        # Identifier et analyser les produits incertains sur test
        uncertain_products = self.identify_uncertain_products(
            df_test, y_pred_test, conf_test, y_true_test, confidence_threshold
        )
        
        # Analyse des produits incertains
        self.analyze_uncertain_products(uncertain_products, df_test)
        
        # Sauvegarder le modèle
        model_path = Path(__file__).parent / 'flat_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'embedding_model_name': 'paraphrase-multilingual-MiniLM-L12-v2'
            }, f)
        print(f"\n💾 Modèle sauvegardé: {model_path}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap_acc,
            'y_true': y_true_test,
            'y_pred': y_pred_test,
            'uncertain_products': uncertain_products
        }
    
    def identify_uncertain_products(self, df, predictions, confidence_scores, y_true, threshold=0.5, output_dir='uncertain_products_flat'):
        """Identifie et sauvegarde les produits incertains en JSON"""
        print(f"\n🔍 Identification des produits incertains (confiance < {threshold})...")
        
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        uncertain_mask = confidence_scores < threshold
        uncertain_data = []
        
        # Construire le mapping catégorie -> path depuis les données
        cat_to_path = dict(zip(df['category_id'], df['category_path']))
        
        for idx in np.where(uncertain_mask)[0]:
            row = df.iloc[idx]
            uncertain_data.append({
                'product_id': row['product_id'],
                'title': str(row['title'])[:100] if pd.notna(row['title']) else '',
                'brand': str(row['brand']) if pd.notna(row['brand']) else '',
                'color': str(row['color']) if pd.notna(row['color']) else '',
                'true_category': y_true[idx],
                'true_path': row['category_path'],
                'predicted_category': predictions[idx],
                'predicted_path': cat_to_path.get(predictions[idx], 'N/A'),
                'confidence_score': float(confidence_scores[idx])
            })
        
        uncertain_data.sort(key=lambda x: x['confidence_score'])
        
        output_file = output_path / 'uncertain_products.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_uncertain_products': len(uncertain_data),
                'products': uncertain_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(uncertain_data)} produits incertains identifiés")
        print(f"   ✓ Sauvegardés dans: {output_file}")
        
        return uncertain_data
    
    def analyze_uncertain_products(self, uncertain_products, df_test):
        """Analyse succincte des produits incertains"""
        print("\n📊 Analyse des produits incertains:")
        
        if len(uncertain_products) == 0:
            print("   Aucun produit incertain")
            return
        
        # Top catégories avec produits incertains
        uncertain_cats = [p['true_category'] for p in uncertain_products]
        cat_counts = pd.Series(uncertain_cats).value_counts().head(5)
        
        print(f"\n   Top 5 catégories avec produits incertains:")
        for cat, count in cat_counts.items():
            pct = (count / len(uncertain_products)) * 100
            print(f"   {cat}: {count} produits ({pct:.1f}%)")
        
        # Brands les plus fréquentes
        brands = [p['brand'] for p in uncertain_products if p['brand']]
        if brands:
            brand_counts = pd.Series(brands).value_counts().head(3)
            print(f"\n   Top 3 brands dans produits incertains:")
            for brand, count in brand_counts.items():
                print(f"   {brand}: {count} produits")
        
        # Confiance moyenne par catégorie (top 3)
        cat_conf = {}
        for p in uncertain_products:
            cat = p['true_category']
            if cat not in cat_conf:
                cat_conf[cat] = []
            cat_conf[cat].append(p['confidence_score'])
        
        avg_conf_by_cat = {cat: np.mean(scores) for cat, scores in cat_conf.items()}
        top_low_conf = sorted(avg_conf_by_cat.items(), key=lambda x: x[1])[:3]
        
        print(f"\n   Top 3 catégories avec confiance moyenne la plus faible:")
        for cat, avg_conf in top_low_conf:
            print(f"   {cat}: {avg_conf:.3f}")
        
        # Pattern : produits avec title très court
        short_titles = [p for p in uncertain_products if len(p['title']) < 20]
        if short_titles:
            print(f"\n   ⚠️  {len(short_titles)} produits avec titre très court (< 20 caractères)")
        
        # Pattern : produits sans brand
        no_brand = [p for p in uncertain_products if not p['brand']]
        if no_brand:
            print(f"   ⚠️  {len(no_brand)} produits sans brand ({len(no_brand)/len(uncertain_products)*100:.1f}%)")


def main():
    """Point d'entrée principal"""
    print("\n" + "="*60)
    print("CLASSIFICATION FLAT")
    print("="*60)
    base_path = Path(__file__).parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    # Vérifier que les fichiers existent
    if not train_path.exists():
        print(f"❌ Fichier non trouvé: {train_path}")
        return
    if not test_path.exists():
        print(f"❌ Fichier non trouvé: {test_path}")
        return
    
    print("\n🔄 Entraînement du classifieur...")
    # Créer et entraîner le classifieur
    classifier = FlatClassifier()
    classifier.train(train_path)
    
    # Évaluer sur train et test
    results = classifier.evaluate(train_path, test_path, confidence_threshold=0.5)
    
    print("\n" + "="*60)
    print("✓ Classification Flat terminée")
    print("="*60)
    
    return classifier, results


if __name__ == '__main__':
    classifier, results = main()

