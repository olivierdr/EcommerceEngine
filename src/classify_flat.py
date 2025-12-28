"""
Classification Flat (Baseline)
Prédiction directe de la catégorie feuille parmi les 100 classes
"""

import pandas as pd
import numpy as np
import hashlib
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class FlatClassifier:
    """Classifieur flat pour prédire directement les catégories feuilles"""
    
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialise le modèle d'embeddings"""
        print("✓ Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.classifier = None
        self.label_encoder = None
        self.df_train = None
        self.X_train = None  # Cache des embeddings d'entraînement
        self.cat_to_path = {}
        
    def prepare_features(self, df, show_progress=True, cache_path=None):
        """Prépare les features textuelles uniquement (title + description)"""
        # Vérifier le cache si un chemin est fourni
        if cache_path and Path(cache_path).exists():
            if show_progress:
                print(f"Chargement des embeddings depuis le cache...")
            return np.load(cache_path)
        
        if show_progress:
            print("Préparation des features...")
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        text_embeddings = self.embedding_model.encode(
            texts.tolist(), 
            show_progress_bar=show_progress,
            batch_size=128
        )
        
        # Sauvegarder dans le cache si un chemin est fourni
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, text_embeddings)
            if show_progress:
                print(f"   ✓ Cache sauvegardé: {cache_path}")
        
        if show_progress:
            print(f"   ✓ {len(texts):,} textes encodés en vecteurs de dimension {text_embeddings.shape[1]}")
        return text_embeddings
    
    def train(self, train_path):
        """Entraîne le modèle sur le train set"""
        print("\n" + "="*60)
        print("ENTRAÎNEMENT - Classification Flat")
        print("="*60)
        
        # Charger les données
        print("\nChargement des données d'entraînement...")
        df_train = pd.read_csv(train_path)
        print(f"   ✓ {len(df_train):,} produits chargés")
        
        # Stocker df_train pour éviter la relecture dans evaluate()
        self.df_train = df_train
        
        # Préparer les features et les stocker pour réutilisation
        self.X_train = self.prepare_features(df_train, show_progress=True)
        y_train = df_train['category_id'].values
        
        # Encoder les labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\nStatistiques:")
        print(f"   Nombre de catégories: {len(self.label_encoder.classes_)}")
        print(f"   Dimension des embeddings: {self.X_train.shape[1]}")
        
        # Entraîner le classifieur
        print("\nEntraînement du classifieur (Logistic Regression)...")
        self.classifier = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.classifier.fit(self.X_train, y_train_encoded)
        print("   ✓ Modèle entraîné")
        
        # Construire cat_to_path pour la production
        self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
        
        return self
    
    def predict(self, df):
        """Prédit les catégories pour un dataframe"""
        X = self.prepare_features(df, show_progress=False)
        y_pred_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, df):
        """Retourne les probabilités de prédiction"""
        X = self.prepare_features(df, show_progress=False)
        return self.classifier.predict_proba(X)
    
    def predict_with_confidence(self, df):
        """Prédit avec scores de confiance"""
        X = self.prepare_features(df, show_progress=False)
        y_pred_encoded = self.classifier.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = self.classifier.predict_proba(X)
        return y_pred, np.max(y_pred_proba, axis=1)
    
    def evaluate(self, train_path=None, test_path=None, confidence_threshold=0.5, df_train=None, df_test=None):
        """Évalue le modèle sur train et test pour détecter le sur-apprentissage"""
        print("\n" + "="*60)
        print("ÉVALUATION - Classification Flat")
        print("="*60)
        
        # Évaluation sur train : réutiliser les embeddings si disponibles
        if df_train is None:
            if hasattr(self, 'df_train') and self.df_train is not None:
                df_train = self.df_train
                print("\nÉvaluation sur données d'entraînement...")
            else:
                if train_path is None:
                    raise ValueError("train_path ou df_train doit être fourni")
                print("\nÉvaluation sur données d'entraînement...")
                df_train = pd.read_csv(train_path)
        
        # Réutiliser X_train si disponible, sinon recalculer
        if self.X_train is not None:
            print("   ✓ Réutilisation des embeddings d'entraînement")
            y_pred_encoded = self.classifier.predict(self.X_train)
            y_pred_train = self.label_encoder.inverse_transform(y_pred_encoded)
            y_pred_proba = self.classifier.predict_proba(self.X_train)
            conf_train = np.max(y_pred_proba, axis=1)
        else:
            y_pred_train, conf_train = self.predict_with_confidence(df_train)
        
        y_true_train = df_train['category_id'].values
        
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            y_true_train, y_pred_train, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
        print(f"   Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1: {train_f1:.4f}")
        
        # Évaluation sur test : utiliser le cache si disponible
        if df_test is None:
            if test_path is None:
                raise ValueError("test_path ou df_test doit être fourni")
            print("\nÉvaluation sur données de test...")
            df_test = pd.read_csv(test_path)
        
        # Générer un hash du chemin pour le cache
        cache_dir = Path(__file__).parent.parent / '.cache'
        cache_dir.mkdir(exist_ok=True)
        if test_path:
            cache_hash = hashlib.md5(str(test_path).encode()).hexdigest()
            cache_path = cache_dir / f'test_embeddings_{cache_hash}.npy'
        else:
            cache_path = None
        
        # Préparer les features avec cache
        X_test = self.prepare_features(df_test, show_progress=True, cache_path=cache_path)
        y_pred_encoded = self.classifier.predict(X_test)
        y_pred_test = self.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = self.classifier.predict_proba(X_test)
        conf_test = np.max(y_pred_proba, axis=1)
        y_true_test = df_test['category_id'].values
        
        test_acc = accuracy_score(y_true_test, y_pred_test)
        test_prec, test_rec, test_f1, _ = precision_recall_fscore_support(
            y_true_test, y_pred_test, average='weighted', zero_division=0
        )
        
        print(f"   Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"   Precision: {test_prec:.4f} | Recall: {test_rec:.4f} | F1: {test_f1:.4f}")
        
        # Comparaison train vs test
        print("\nComparaison Train vs Test:")
        gap_acc = train_acc - test_acc
        gap_prec = train_prec - test_prec
        gap_rec = train_rec - test_rec
        gap_f1 = train_f1 - test_f1
        
        print(f"   Écart Accuracy: {gap_acc:.4f} ({gap_acc*100:.2f} points)")
        print(f"   Écart Precision: {gap_prec:.4f} | Recall: {gap_rec:.4f} | F1: {gap_f1:.4f}")
        
        if gap_acc > 0.05:
            print(f"Warning: Sur-apprentissage détecté (écart > 5 points)")
        else:
            print(f"✓ Pas de sur-apprentissage significatif")
        
        # Analyses détaillées : catégories certaines, incertaines et patterns de confusion
        self.analyze_categories(df_test, y_pred_test, conf_test, y_true_test, confidence_threshold)
        
        # Sauvegarder le modèle avec métadonnées pour la production
        model_path = Path(__file__).parent.parent / 'results' / 'classification' / 'flat_model.pkl'
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'embedding_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
                'cat_to_path': self.cat_to_path
            }, f)
        print(f"\nModèle sauvegardé: {model_path}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap_acc,
            'y_true': y_true_test,
            'y_pred': y_pred_test
        }
    
    def analyze_categories(self, df, predictions, confidence_scores, y_true, threshold=0.5):
        """Analyse unifiée : génère les 3 JSON (certain, uncertain, confusion)"""
        print("\n" + "="*60)
        print("ANALYSES DÉTAILLÉES")
        print("="*60)
        
        # Charger les noms de catégories
        names_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        category_names = {}
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(list(data.values())[0], dict):
                    category_names = {cat_id: data[cat_id]['name'] for cat_id in data}
                else:
                    category_names = data
        
        # Préparer les données
        df_analysis = df.copy()
        df_analysis['predicted_category'] = predictions
        df_analysis['confidence'] = confidence_scores
        df_analysis['is_certain'] = confidence_scores >= threshold
        df_analysis['is_correct'] = predictions == y_true
        
        cat_to_path = dict(zip(df['category_id'], df['category_path']))
        
        # Initialiser les structures
        certain_stats = []
        uncertain_stats = []
        confusion_pairs = {}
        
        # Parcourir les données une seule fois
        for cat_id in df['category_id'].unique():
            cat_data = df_analysis[df_analysis['category_id'] == cat_id]
            cat_name = category_names.get(cat_id, 'N/A')
            total_in_cat = len(cat_data)
            
            # Certain et Uncertain
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
            
            # Confusion patterns (erreurs uniquement)
            errors = cat_data[cat_data['predicted_category'] != cat_id]
            for _, row in errors.iterrows():
                pred_cat = row['predicted_category']
                pair_key = (cat_id, pred_cat)
                
                if pair_key not in confusion_pairs:
                    confusion_pairs[pair_key] = {
                        'confidences': [],
                        'examples': []
                    }
                
                confusion_pairs[pair_key]['confidences'].append(row['confidence'])
                if len(confusion_pairs[pair_key]['examples']) < 3:
                    confusion_pairs[pair_key]['examples'].append({
                        'product_id': row['product_id'],
                        'title': str(row['title'])[:100] if pd.notna(row['title']) else ''
                    })
        
        # Trier et sauvegarder Certain
        certain_stats.sort(key=lambda x: x['n_certain_products'], reverse=True)
        certain_df = df_analysis[df_analysis['is_certain']]
        output_path_certain = Path(__file__).parent.parent / 'results' / 'classification' / 'certain_categories_analysis.json'
        output_path_certain.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_certain, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_certain_products': len(certain_df),
                    'total_categories_with_certain': len(certain_stats),
                    'avg_confidence': round(float(np.mean(certain_df['confidence'])), 3),
                    'threshold': threshold
                },
                'top_10_categories': certain_stats[:10]
            }, f, indent=2, ensure_ascii=False)
        
        # Trier et sauvegarder Uncertain
        uncertain_stats.sort(key=lambda x: x['n_uncertain_products'], reverse=True)
        uncertain_df = df_analysis[~df_analysis['is_certain']]
        output_path_uncertain = Path(__file__).parent.parent / 'results' / 'classification' / 'uncertain_categories_analysis.json'
        output_path_uncertain.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_uncertain, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_uncertain_products': len(uncertain_df),
                    'total_categories_with_uncertain': len(uncertain_stats),
                    'avg_confidence': round(float(np.mean(uncertain_df['confidence'])), 3),
                    'threshold': threshold
                },
                'top_10_categories': uncertain_stats[:10]
            }, f, indent=2, ensure_ascii=False)
        
        # Calculer et sauvegarder Confusion Patterns
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
        output_path_confusion = Path(__file__).parent.parent / 'results' / 'classification' / 'confusion_patterns.json'
        output_path_confusion.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_confusion, 'w', encoding='utf-8') as f:
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
    print("CLASSIFICATION FLAT")
    print("="*60)
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    # Vérifier que les fichiers existent
    if not train_path.exists():
        print(f"Fichier non trouvé: {train_path}")
        return
    if not test_path.exists():
        print(f"Fichier non trouvé: {test_path}")
        return
    
    print("\nEntraînement du classifieur...")
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

