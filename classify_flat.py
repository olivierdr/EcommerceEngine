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
        print("🔄 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.classifier = None
        self.label_encoder = None
        self.df_train = None
        self.X_train = None  # Cache des embeddings d'entraînement
        self.cat_to_path = {}
        self.problematic_categories = set()
        
    def prepare_features(self, df, show_progress=True, cache_path=None):
        """Prépare les features textuelles uniquement (title + description)"""
        # Vérifier le cache si un chemin est fourni
        if cache_path and Path(cache_path).exists():
            if show_progress:
                print(f"📝 Chargement des embeddings depuis le cache...")
            return np.load(cache_path)
        
        if show_progress:
            print("📝 Préparation des features...")
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
        print("🚀 ENTRAÎNEMENT - Classification Flat")
        print("="*60)
        
        # Charger les données
        print("\n📊 Chargement des données d'entraînement...")
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
        
        print(f"\n📈 Statistiques:")
        print(f"   Nombre de catégories: {len(self.label_encoder.classes_)}")
        print(f"   Dimension des embeddings: {self.X_train.shape[1]}")
        
        # Entraîner le classifieur
        print("\n🎯 Entraînement du classifieur (Logistic Regression)...")
        self.classifier = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.classifier.fit(self.X_train, y_train_encoded)
        print("   ✓ Modèle entraîné")
        
        # Construire cat_to_path pour la production
        self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
        
        # Identifier les catégories problématiques (à partir des prédictions d'entraînement)
        print("\n🔍 Identification des catégories problématiques...")
        y_pred_train, conf_train = self.predict_with_confidence(df_train)
        problematic_categories = self._identify_problematic_categories(df_train, y_pred_train, conf_train)
        self.problematic_categories = problematic_categories
        print(f"   ✓ {len(problematic_categories)} catégories problématiques identifiées")
        
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
        print("📊 ÉVALUATION - Classification Flat")
        print("="*60)
        
        # Évaluation sur train : réutiliser les embeddings si disponibles
        if df_train is None:
            if hasattr(self, 'df_train') and self.df_train is not None:
                df_train = self.df_train
                print("\n📊 Évaluation sur données d'entraînement (réutilisation des données)...")
            else:
                if train_path is None:
                    raise ValueError("train_path ou df_train doit être fourni")
                print("\n📊 Évaluation sur données d'entraînement...")
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
            print("\n📊 Évaluation sur données de test...")
            df_test = pd.read_csv(test_path)
        
        # Générer un hash du chemin pour le cache
        cache_dir = Path(__file__).parent / '.cache'
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
        
        # Analyses détaillées : catégories certaines, incertaines et patterns de confusion
        self.generate_detailed_analyses(df_test, y_pred_test, conf_test, y_true_test, confidence_threshold)
        
        # Sauvegarder le modèle avec métadonnées pour la production
        model_path = Path(__file__).parent / 'flat_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'embedding_model_name': 'paraphrase-multilingual-MiniLM-L12-v2',
                'cat_to_path': self.cat_to_path,
                'problematic_categories': self.problematic_categories
            }, f)
        print(f"\n💾 Modèle sauvegardé: {model_path}")
        
        return {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'gap': gap_acc,
            'y_true': y_true_test,
            'y_pred': y_pred_test
        }
    
    def load_category_names(self):
        """Charge les noms de catégories depuis category_names.json"""
        names_path = Path(__file__).parent / 'category_names.json'
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Adapter au format actuel (dict avec 'name' et 'example_titles')
                if isinstance(list(data.values())[0], dict):
                    return {cat_id: data[cat_id]['name'] for cat_id in data}
                else:
                    return data
        return {}
    
    def generate_detailed_analyses(self, df, predictions, confidence_scores, y_true, threshold=0.5):
        """Génère les analyses détaillées : catégories certaines, incertaines et patterns de confusion"""
        print("\n" + "="*60)
        print("📊 GÉNÉRATION DES ANALYSES DÉTAILLÉES")
        print("="*60)
        
        category_names = self.load_category_names()
        
        # Créer un DataFrame avec toutes les infos
        df_analysis = df.copy()
        df_analysis['predicted_category'] = predictions
        df_analysis['confidence'] = confidence_scores
        df_analysis['is_certain'] = confidence_scores >= threshold
        df_analysis['is_correct'] = predictions == y_true
        
        # 1. Analyse des catégories certaines
        self.analyze_certain_categories(df_analysis, category_names, threshold)
        
        # 2. Analyse des catégories incertaines
        self.analyze_uncertain_categories(df_analysis, category_names, threshold)
        
        # 3. Analyse des patterns de confusion
        self.analyze_confusion_patterns(df_analysis, category_names)
    
    def analyze_certain_categories(self, df, category_names, threshold):
        """Analyse des catégories avec produits certains (confiance >= threshold)"""
        print("\n📈 Analyse des catégories certaines...")
        
        certain_df = df[df['is_certain']]
        
        if len(certain_df) == 0:
            print("   Aucun produit certain")
            return
        
        # Grouper par catégorie vraie
        cat_stats = []
        for cat_id in certain_df['category_id'].unique():
            cat_data = certain_df[certain_df['category_id'] == cat_id]
            cat_name = category_names.get(cat_id, 'N/A')
            
            # Calculer les métriques
            confidences = cat_data['confidence'].values
            titles = cat_data['title'].fillna('').astype(str)
            
            # Construire le mapping catégorie -> path
            cat_to_path = dict(zip(df['category_id'], df['category_path']))
            
            # Exemples avec infos complètes
            example_titles = []
            for idx in cat_data.head(5).index:
                row = cat_data.loc[idx]
                example_titles.append({
                    'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
                    'confidence': round(float(row['confidence']), 3),
                    'true_category': cat_id,
                    'true_path': row['category_path'],
                    'predicted_category': row['predicted_category'],
                    'predicted_path': cat_to_path.get(row['predicted_category'], 'N/A')
                })
            
            cat_stats.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'n_certain_products': len(cat_data),
                'avg_confidence': round(float(np.mean(confidences)), 3),
                'certainty_rate': round(len(cat_data) / len(df[df['category_id'] == cat_id]), 3),
                'avg_title_length': round(float(np.mean([len(t) for t in titles])), 3),
                'brand_presence_rate': round(float(cat_data['brand'].notna().sum() / len(cat_data)), 3),
                'color_presence_rate': round(float(cat_data['color'].notna().sum() / len(cat_data)), 3),
                'example_titles': example_titles
            })
        
        # Trier par nombre de produits certains (desc)
        cat_stats.sort(key=lambda x: x['n_certain_products'], reverse=True)
        
        # Top 10
        top_10 = cat_stats[:10]
        
        summary = {
            'total_certain_products': len(certain_df),
            'total_categories_with_certain': len(cat_stats),
            'avg_confidence': round(float(np.mean(certain_df['confidence'])), 3),
            'threshold': threshold
        }
        
        output = {
            'summary': summary,
            'top_10_categories': top_10
        }
        
        output_path = Path(__file__).parent / 'certain_categories_analysis.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(cat_stats)} catégories avec produits certains")
        print(f"   ✓ Top 10 sauvegardés dans: {output_path}")
    
    def analyze_uncertain_categories(self, df, category_names, threshold):
        """Analyse des catégories avec produits incertains (confiance < threshold)"""
        print("\n📉 Analyse des catégories incertaines...")
        
        uncertain_df = df[df['confidence'] < threshold]
        
        if len(uncertain_df) == 0:
            print("   Aucun produit incertain")
            return
        
        # Construire le mapping catégorie -> path
        cat_to_path = dict(zip(df['category_id'], df['category_path']))
        
        # Grouper par catégorie vraie
        cat_stats = []
        for cat_id in uncertain_df['category_id'].unique():
            cat_data = uncertain_df[uncertain_df['category_id'] == cat_id]
            cat_name = category_names.get(cat_id, 'N/A')
            
            # Total de produits dans cette catégorie
            total_in_cat = len(df[df['category_id'] == cat_id])
            
            # Calculer les métriques
            confidences = cat_data['confidence'].values
            titles = cat_data['title'].fillna('').astype(str)
            
            # Exemples avec infos complètes
            example_titles = []
            for idx in cat_data.head(5).index:
                row = cat_data.loc[idx]
                example_titles.append({
                    'title': str(row['title'])[:80] if pd.notna(row['title']) else '',
                    'confidence': round(float(row['confidence']), 3),
                    'true_category': cat_id,
                    'true_path': row['category_path'],
                    'predicted_category': row['predicted_category'],
                    'predicted_path': cat_to_path.get(row['predicted_category'], 'N/A')
                })
            
            cat_stats.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'n_uncertain_products': len(cat_data),
                'uncertainty_rate': round(len(cat_data) / total_in_cat, 3),
                'avg_confidence': round(float(np.mean(confidences)), 3),
                'avg_title_length': round(float(np.mean([len(t) for t in titles])), 3),
                'brand_presence_rate': round(float(cat_data['brand'].notna().sum() / len(cat_data)), 3),
                'color_presence_rate': round(float(cat_data['color'].notna().sum() / len(cat_data)), 3),
                'example_titles': example_titles
            })
        
        # Trier par nombre de produits incertains (desc)
        cat_stats.sort(key=lambda x: x['n_uncertain_products'], reverse=True)
        
        # Top 10
        top_10 = cat_stats[:10]
        
        summary = {
            'total_uncertain_products': len(uncertain_df),
            'total_categories_with_uncertain': len(cat_stats),
            'avg_confidence': round(float(np.mean(uncertain_df['confidence'])), 3),
            'threshold': threshold
        }
        
        output = {
            'summary': summary,
            'top_10_categories': top_10
        }
        
        output_path = Path(__file__).parent / 'uncertain_categories_analysis.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(cat_stats)} catégories avec produits incertains")
        print(f"   ✓ Top 10 sauvegardés dans: {output_path}")
    
    def analyze_confusion_patterns(self, df, category_names):
        """Analyse des patterns de confusion (catégories confondues)"""
        print("\n🔄 Analyse des patterns de confusion...")
        
        # Filtrer les erreurs uniquement
        errors = df[df['predicted_category'] != df['category_id']]
        
        if len(errors) == 0:
            print("   Aucune erreur détectée")
            return
        
        # Compter les paires de confusion
        confusion_pairs = {}
        for _, row in errors.iterrows():
            true_cat = row['category_id']
            pred_cat = row['predicted_category']
            pair_key = (true_cat, pred_cat)
            
            if pair_key not in confusion_pairs:
                confusion_pairs[pair_key] = {
                    'confidences': [],
                    'examples': []
                }
            
            confusion_pairs[pair_key]['confidences'].append(row['confidence'])
            # Garder max 3 exemples par paire
            if len(confusion_pairs[pair_key]['examples']) < 3:
                confusion_pairs[pair_key]['examples'].append({
                    'product_id': row['product_id'],
                    'title': str(row['title'])[:100] if pd.notna(row['title']) else ''
                })
        
        # Calculer les statistiques pour chaque paire
        patterns = []
        for (true_cat, pred_cat), data in confusion_pairs.items():
            true_name = category_names.get(true_cat, 'N/A')
            pred_name = category_names.get(pred_cat, 'N/A')
            
            # Taux de confusion : erreurs / total produits de la catégorie vraie
            total_true_cat = len(df[df['category_id'] == true_cat])
            confusion_rate = len(data['confidences']) / total_true_cat
            
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
        
        # Trier par nombre de cas (desc)
        patterns.sort(key=lambda x: x['n_cases'], reverse=True)
        
        # Top 10
        top_10 = patterns[:10]
        
        summary = {
            'total_confusion_cases': len(errors),
            'unique_confusion_pairs': len(patterns)
        }
        
        output = {
            'summary': summary,
            'top_10_confusion_patterns': top_10
        }
        
        output_path = Path(__file__).parent / 'confusion_patterns.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(patterns)} patterns de confusion identifiés")
        print(f"   ✓ Top 10 sauvegardés dans: {output_path}")
    
    def _identify_problematic_categories(self, df, predictions, confidence_scores, threshold=0.5, min_uncertainty_rate=0.4):
        """Identifie les catégories problématiques basées sur le taux d'incertitude"""
        problematic = set()
        uncertain_df = df[confidence_scores < threshold].copy()
        uncertain_df['predicted_category'] = predictions[confidence_scores < threshold]
        
        for cat_id in uncertain_df['category_id'].unique():
            cat_uncertain = uncertain_df[uncertain_df['category_id'] == cat_id]
            total_in_cat = len(df[df['category_id'] == cat_id])
            uncertainty_rate = len(cat_uncertain) / total_in_cat if total_in_cat > 0 else 0
            
            if uncertainty_rate >= min_uncertainty_rate:
                problematic.add(cat_id)
        
        return problematic


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

