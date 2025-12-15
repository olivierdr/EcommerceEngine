"""
Classification Hiérarchique Top-Down
Prédiction niveau par niveau jusqu'à la catégorie feuille
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class HierarchicalClassifier:
    """Classifieur hiérarchique top-down"""
    
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialise le modèle d'embeddings"""
        print("🔄 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.hierarchy = {}
        self.models = {}
        self.label_encoders = {}
        self.max_depth = 0
        self.train_path = None
        self.cat_to_path = {}
        
    def build_hierarchy(self, df):
        """Construit la structure hiérarchique depuis les category_path"""
        print("🌳 Construction de la hiérarchie...")
        
        hierarchy = defaultdict(lambda: {'children': set(), 'parent': None, 'level': 0})
        
        for _, row in df.iterrows():
            path = row['category_path'].split('/')
            depth = len(path)
            self.max_depth = max(self.max_depth, depth)
            
            # Construire la hiérarchie niveau par niveau
            for level in range(depth):
                node_id = path[level]
                hierarchy[node_id]['level'] = level + 1
                
                if level > 0:
                    parent_id = path[level - 1]
                    hierarchy[node_id]['parent'] = parent_id
                    hierarchy[parent_id]['children'].add(node_id)
        
        # Convertir les sets en listes pour la sérialisation
        self.hierarchy = {
            node: {
                'children': sorted(list(data['children'])),
                'parent': data['parent'],
                'level': data['level']
            }
            for node, data in hierarchy.items()
        }
        
        print(f"   ✓ Hiérarchie construite (profondeur max: {self.max_depth})")
        return self.hierarchy
    
    def prepare_features(self, df):
        """Prépare les features textuelles"""
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        embeddings = self.embedding_model.encode(texts.tolist(), show_progress_bar=False)
        return embeddings
    
    def get_level_labels(self, df, level):
        """Extrait les labels pour un niveau donné"""
        labels = []
        for _, row in df.iterrows():
            path = row['category_path'].split('/')
            if len(path) >= level:
                labels.append(path[level - 1])
            else:
                labels.append(None)
        return np.array(labels)
    
    def train(self, train_path):
        """Entraîne les modèles par niveau"""
        print("\n" + "="*60)
        print("🚀 ENTRAÎNEMENT - Classification Hiérarchique")
        print("="*60)
        
        # Stocker le chemin pour l'évaluation
        self.train_path = train_path
        
        # Charger les données
        print("\n📊 Chargement des données d'entraînement...")
        df_train = pd.read_csv(train_path)
        print(f"   ✓ {len(df_train):,} produits chargés")
        
        # Construire le mapping catégorie -> path
        self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
        
        # Construire la hiérarchie
        self.build_hierarchy(df_train)
        
        # Préparer les features (une seule fois)
        print("\n📝 Préparation des features...")
        X_train = self.prepare_features(df_train)
        
        # Entraîner un modèle pour chaque niveau
        print("\n🎯 Entraînement des modèles par niveau...")
        
        for level in range(1, self.max_depth + 1):
            # Extraire les labels pour ce niveau
            y_level = self.get_level_labels(df_train, level)
            
            # Filtrer les cas valides (pas de None)
            valid_mask = y_level != None
            if valid_mask.sum() == 0:
                continue
            
            X_level = X_train[valid_mask]
            y_level_valid = y_level[valid_mask]
            
            # Pour les niveaux > 1, on entraîne des modèles conditionnels
            # (un modèle par parent pour prédire parmi ses enfants)
            if level == 1:
                # Niveau 1 : un seul modèle (1 seule catégorie racine)
                unique_labels = np.unique(y_level_valid)
                if len(unique_labels) == 1:
                    # Cas trivial : une seule catégorie
                    self.models[level] = {'root': unique_labels[0]}
                    self.label_encoders[level] = {'root': None}
                    print(f"   Niveau {level}: 1 catégorie (trivial)")
                    continue
            
            # Pour les autres niveaux, créer un modèle par parent
            models_level = {}
            encoders_level = {}
            
            # Grouper par parent
            parent_groups = defaultdict(list)
            for idx, (x, y) in enumerate(zip(X_level, y_level_valid)):
                # Trouver le parent
                parent = self.hierarchy[y]['parent'] if y in self.hierarchy else None
                if parent is None and level > 1:
                    continue
                parent_key = parent if parent else 'root'
                parent_groups[parent_key].append((idx, y))
            
            # Entraîner un modèle pour chaque parent
            for parent_key, samples in parent_groups.items():
                if len(samples) == 0:
                    continue
                
                indices = [idx for idx, _ in samples]
                labels = [label for _, label in samples]
                
                unique_labels = np.unique(labels)
                if len(unique_labels) == 1:
                    # Cas trivial : un seul enfant
                    models_level[parent_key] = unique_labels[0]
                    encoders_level[parent_key] = None
                else:
                    # Modèle de classification
                    encoder = LabelEncoder()
                    y_encoded = encoder.fit_transform(labels)
                    
                    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                    model.fit(X_level[indices], y_encoded)
                    
                    models_level[parent_key] = model
                    encoders_level[parent_key] = encoder
            
            self.models[level] = models_level
            self.label_encoders[level] = encoders_level
            
            n_parents = len(models_level)
            print(f"   Niveau {level}: {n_parents} modèles conditionnels entraînés")
        
        print("   ✓ Tous les modèles entraînés")
        return self
    
    def predict(self, df):
        """Prédit les catégories feuilles niveau par niveau"""
        X = self.prepare_features(df)
        predictions = []
        
        for idx in range(len(df)):
            # Prédiction niveau par niveau
            current_parent = None
            predicted = None
            
            for level in range(1, self.max_depth + 1):
                if level not in self.models:
                    break
                
                models_level = self.models[level]
                encoders_level = self.label_encoders[level]
                
                # Déterminer la clé du parent
                parent_key = current_parent if current_parent else 'root'
                
                # Vérifier si on a un modèle pour ce parent
                if parent_key not in models_level:
                    break
                
                model_or_label = models_level[parent_key]
                encoder = encoders_level[parent_key]
                
                # Cas trivial (un seul enfant)
                if encoder is None:
                    predicted = model_or_label
                else:
                    # Prédiction avec le modèle
                    x_sample = X[idx:idx+1]
                    y_pred_encoded = model_or_label.predict(x_sample)[0]
                    predicted = encoder.inverse_transform([y_pred_encoded])[0]
                
                # Vérifier si c'est une feuille (pas d'enfants)
                if predicted in self.hierarchy:
                    children = self.hierarchy[predicted]['children']
                    if len(children) == 0:
                        # C'est une feuille → c'est notre prédiction finale
                        break
                    else:
                        # Continuer avec le niveau suivant
                        current_parent = predicted
                else:
                    # Catégorie non trouvée → arrêter
                    break
            
            # Utiliser la dernière prédiction comme catégorie feuille
            if predicted:
                predictions.append(predicted)
            else:
                # Fallback : utiliser la première catégorie trouvée
                predictions.append(list(self.hierarchy.keys())[0])
        
        return np.array(predictions)
    
    def evaluate(self, test_path):
        """Évalue le modèle sur le test set"""
        print("\n" + "="*60)
        print("📊 ÉVALUATION - Classification Hiérarchique")
        print("="*60)
        
        # Charger les données de test
        print("\n📊 Chargement des données de test...")
        df_test = pd.read_csv(test_path)
        print(f"   ✓ {len(df_test):,} produits chargés")
        
        # Prédictions
        print("\n🔮 Prédictions...")
        y_pred = self.predict(df_test)
        y_true = df_test['category_id'].values
        
        # Accuracy finale (catégorie feuille)
        accuracy_final = accuracy_score(y_true, y_pred)
        print(f"\n✅ Accuracy finale (catégorie feuille): {accuracy_final:.4f} ({accuracy_final*100:.2f}%)")
        
        # Accuracy par niveau (simplifié : on compare les paths complets)
        print("\n📊 Accuracy par niveau (basé sur les paths complets):")
        
        accuracies_by_level = {}
        for level in range(1, min(self.max_depth + 1, 6)):  # Limiter aux 5 premiers niveaux
            correct = 0
            total = 0
            
            for idx, (true_cat, pred_cat) in enumerate(zip(y_true, y_pred)):
                true_path = df_test.iloc[idx]['category_path'].split('/')
                
                # Trouver le path de la catégorie prédite
                if pred_cat in self.cat_to_path:
                    pred_path = self.cat_to_path[pred_cat].split('/')
                else:
                    continue
                
                if len(true_path) >= level and len(pred_path) >= level:
                    total += 1
                    if true_path[level - 1] == pred_path[level - 1]:
                        correct += 1
            
            if total > 0:
                acc_level = correct / total
                accuracies_by_level[level] = acc_level
                print(f"   Niveau {level}: {acc_level:.4f} ({acc_level*100:.2f}%) - {correct}/{total}")
        
        # Analyse des erreurs
        errors = y_true != y_pred
        error_rate = errors.sum() / len(y_true)
        print(f"\n📉 Taux d'erreur: {error_rate:.4f} ({error_rate*100:.2f}%)")
        print(f"   Erreurs: {errors.sum()} / {len(y_true)}")
        
        return {
            'accuracy': accuracy_final,
            'y_true': y_true,
            'y_pred': y_pred,
            'error_rate': error_rate
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
    classifier = HierarchicalClassifier()
    classifier.train(train_path)
    
    # Évaluer sur le test set
    results = classifier.evaluate(test_path)
    
    print("\n" + "="*60)
    print("✓ Classification Hiérarchique terminée")
    print("="*60)
    
    return classifier, results


if __name__ == '__main__':
    classifier, results = main()

