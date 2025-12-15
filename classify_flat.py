"""
Classification Flat (Baseline)
Prédiction directe de la catégorie feuille parmi les 100 classes
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
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
        """Prépare les features textuelles (title + description)"""
        print("📝 Préparation des features...")
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        embeddings = self.embedding_model.encode(texts.tolist(), show_progress_bar=False)
        return embeddings
    
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
    
    def evaluate(self, test_path):
        """Évalue le modèle sur le test set"""
        print("\n" + "="*60)
        print("📊 ÉVALUATION - Classification Flat")
        print("="*60)
        
        # Charger les données de test
        print("\n📊 Chargement des données de test...")
        df_test = pd.read_csv(test_path)
        print(f"   ✓ {len(df_test):,} produits chargés")
        
        # Prédictions
        print("\n🔮 Prédictions...")
        y_pred = self.predict(df_test)
        y_true = df_test['category_id'].values
        
        # Métriques principales
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n✅ Accuracy globale: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report (top classes)
        print("\n📋 Rapport de classification (top 10 catégories par support):")
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Afficher les top catégories
        category_counts = pd.Series(y_true).value_counts().head(10)
        print("\n   Top catégories:")
        for cat_id in category_counts.index:
            if cat_id in report:
                prec = report[cat_id]['precision']
                rec = report[cat_id]['recall']
                f1 = report[cat_id]['f1-score']
                support = report[cat_id]['support']
                print(f"   {cat_id}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (n={support})")
        
        # Résumé global
        print(f"\n📊 Résumé global:")
        print(f"   Precision moyenne: {report['weighted avg']['precision']:.4f}")
        print(f"   Recall moyen: {report['weighted avg']['recall']:.4f}")
        print(f"   F1-score moyen: {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'y_true': y_true,
            'y_pred': y_pred,
            'report': report
        }


def main():
    """Point d'entrée principal"""
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
    
    # Créer et entraîner le classifieur
    classifier = FlatClassifier()
    classifier.train(train_path)
    
    # Évaluer sur le test set
    results = classifier.evaluate(test_path)
    
    print("\n" + "="*60)
    print("✓ Classification Flat terminée")
    print("="*60)
    
    return classifier, results


if __name__ == '__main__':
    classifier, results = main()

