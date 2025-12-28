"""
Étape 1 - Audit de la taxonomie existante

Ce script analyse le jeu de données d'entraînement pour :
1. Comprendre la structure de la taxonomie (profondeur, nombre de catégories par niveau)
2. Détecter des incohérences structurelles dans les category_path
3. Évaluer la cohérence sémantique des produits au sein de chaque catégorie
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import warnings
warnings.filterwarnings('ignore')

# Pour les embeddings sémantiques
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence-transformers non installé. L'analyse sémantique sera limitée.")


class TaxonomyAuditor:
    """Auditeur de taxonomie pour produits e-commerce"""
    
    def __init__(self, data_path, sample_size=None):
        """
        Parameters:
        -----------
        data_path : str
            Chemin vers le fichier CSV d'entraînement
        sample_size : int, optional
            Nombre d'échantillons à utiliser pour l'analyse sémantique (None = tout)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.df = None
        self.embedding_model = None
        
    def load_data(self):
        """Charge les données d'entraînement"""
        print("Chargement des données...")
        self.df = pd.read_csv(self.data_path)
        print(f"   ✓ {len(self.df):,} produits chargés")
        return self.df
    
    def analyze_structure(self):
        """Analyse la structure de la taxonomie"""
        print("\n" + "="*60)
        print("1 ANALYSE DE LA STRUCTURE DE LA TAXONOMIE")
        print("="*60)
        
        # Profondeur des chemins
        self.df['path_depth'] = self.df['category_path'].str.count('/') + 1
        depths = self.df['path_depth'].value_counts().sort_index()
        
        print(f"\nProfondeur de la taxonomie:")
        print(f"   Profondeur minimale: {depths.index.min()}")
        print(f"   Profondeur maximale: {depths.index.max()}")
        print(f"   Profondeur moyenne: {self.df['path_depth'].mean():.2f}")
        print(f"   Profondeur médiane: {self.df['path_depth'].median():.0f}")
        
        print(f"\nDistribution des profondeurs:")
        for depth, count in depths.items():
            pct = (count / len(self.df)) * 100
            print(f"   Niveau {depth}: {count:,} produits ({pct:.1f}%)")
        
        # Nombre de catégories uniques par niveau
        print(f"\nNombre de catégories uniques par niveau:")
        max_depth = int(self.df['path_depth'].max())
        for level in range(1, max_depth + 1):
            categories_at_level = set()
            for path in self.df['category_path']:
                parts = path.split('/')
                if len(parts) >= level:
                    categories_at_level.add(parts[level - 1])
            print(f"   Niveau {level}: {len(categories_at_level):,} catégories uniques")
        
        # Catégories feuilles
        unique_leaf_categories = self.df['category_id'].nunique()
        print(f"\nCatégories feuilles (category_id):")
        print(f"   {unique_leaf_categories:,} catégories feuilles uniques")
        print(f"   {len(self.df):,} produits au total")
        avg_products_per_category = len(self.df) / unique_leaf_categories
        print(f"   Moyenne: {avg_products_per_category:.1f} produits par catégorie")
        
        # Distribution des produits par catégorie
        category_counts = self.df['category_id'].value_counts()
        print(f"\nDistribution des produits par catégorie:")
        print(f"   Catégorie la plus fréquente: {category_counts.max():,} produits")
        print(f"   Catégorie la moins fréquente: {category_counts.min():,} produits")
        print(f"   Médiane: {category_counts.median():.0f} produits")
        
        # Top 5 catégories les plus fréquentes
        print(f"\nTop 5 catégories les plus fréquentes:")
        top_5 = category_counts.head(5)
        for cat_id, count in top_5.items():
            print(f"   {cat_id}: {count:,} produits")
        
        # Bottom 5 catégories les moins fréquentes
        print(f"\nTop 5 catégories les moins fréquentes:")
        bottom_5 = category_counts.tail(5)
        for cat_id, count in bottom_5.items():
            print(f"   {cat_id}: {count:,} produits")
        
        # Générer les noms de catégories
        category_names = self.generate_category_names()
        
        return {
            'depths': depths,
            'max_depth': max_depth,
            'unique_leaf_categories': unique_leaf_categories,
            'category_counts': category_counts,
            'category_names': category_names
        }
    
    def generate_category_names(self):
        """Génère des noms simples pour chaque catégorie basés sur les mots-clés fréquents"""
        print("\nGénération des noms de catégories...")
        
        # Stopwords simples (FR/DE/EN)
        stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'pour', 'avec', 'sans', 
                    'der', 'die', 'das', 'und', 'oder', 'für', 'mit', 'ohne',
                    'the', 'a', 'an', 'and', 'or', 'for', 'with', 'without',
                    'à', 'd', 'l', 'un', 'une', 'en', 'sur', 'par', 'dans'}
        
        category_data = {}
        
        for cat_id in self.df['category_id'].unique():
            cat_products = self.df[self.df['category_id'] == cat_id]
            titles = cat_products['title'].fillna('').astype(str).tolist()
            
            # Extraire les mots (min 3 caractères, alphanumériques)
            words = []
            for title in titles:
                title_words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', title.lower())
                words.extend([w for w in title_words if w not in stopwords])
            
            # Top 2-3 mots les plus fréquents
            if words:
                word_counts = Counter(words)
                top_words = [word for word, _ in word_counts.most_common(3)]
                category_name = ' '.join(top_words).title()
            else:
                category_name = "Catégorie inconnue"
            
            # Exemples de titres (max 5)
            example_titles = [t[:80] for t in titles[:5] if t.strip()]
            
            category_data[cat_id] = {
                'name': category_name,
                'example_titles': example_titles
            }
        
        # Sauvegarder
        output_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(category_data)} noms générés")
        print(f"Sauvegardés dans: {output_path}")
        
        return category_data
    
    def load_category_names(self):
        """Charge les noms de catégories depuis category_names.json"""
        names_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def detect_inconsistencies(self):
        """Détecte les incohérences structurelles dans les category_path"""
        print("\n" + "="*60)
        print("2 DÉTECTION D'INCOHÉRENCES STRUCTURELLES")
        print("="*60)
        
        inconsistencies = {
            'category_id_mismatch': [],
            'empty_paths': [],
            'invalid_paths': []
        }
        
        # Vérifier que category_id correspond au dernier élément du path
        print("\nVérification de cohérence category_id / category_path...")
        mismatches = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            last_path_id = path_parts[-1] if path_parts else None
            if last_path_id != row['category_id']:
                mismatches += 1
                inconsistencies['category_id_mismatch'].append({
                    'product_id': row['product_id'],
                    'category_id': row['category_id'],
                    'last_path_id': last_path_id,
                    'category_path': row['category_path']
                })
        
        if mismatches > 0:
            print(f"{mismatches:,} incohérences détectées (category_id ≠ dernier élément du path)")
            print(f"   Exemples (premiers 5):")
            for inc in inconsistencies['category_id_mismatch'][:5]:
                print(f"      - Product: {inc['product_id'][:8]}... | category_id: {inc['category_id']} | path fin: {inc['last_path_id']}")
        else:
            print(f"   ✓ Aucune incohérence détectée")
        
        # Vérifier les paths vides ou invalides
        print("\nVérification des paths vides ou invalides...")
        empty_paths = self.df[self.df['category_path'].isna() | (self.df['category_path'] == '')]
        if len(empty_paths) > 0:
            print(f"{len(empty_paths):,} produits avec path vide")
            inconsistencies['empty_paths'] = empty_paths['product_id'].tolist()
        else:
            print(f"   ✓ Aucun path vide")
        
        # Vérifier les paths avec des IDs invalides (format hexadécimal attendu)
        print("\nVérification du format des IDs dans les paths...")
        invalid_format = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            for part in path_parts:
                # Vérifier que c'est un hexadécimal de 8 caractères
                if len(part) != 8 or not all(c in '0123456789abcdef' for c in part.lower()):
                    invalid_format += 1
                    inconsistencies['invalid_paths'].append({
                        'product_id': row['product_id'],
                        'category_path': row['category_path'],
                        'invalid_part': part
                    })
                    break
        
        if invalid_format > 0:
            print(f"{invalid_format:,} paths avec format d'ID invalide")
        else:
            print(f"   ✓ Tous les IDs ont un format valide (8 caractères hex)")
        
        # Vérifier les chemins avec des doublons consécutifs
        print("\nVérification des doublons consécutifs dans les paths...")
        consecutive_duplicates = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            for i in range(len(path_parts) - 1):
                if path_parts[i] == path_parts[i + 1]:
                    consecutive_duplicates += 1
                    break
        
        if consecutive_duplicates > 0:
            print(f"{consecutive_duplicates:,} paths avec doublons consécutifs")
        else:
            print(f"   ✓ Aucun doublon consécutif")
        
        return inconsistencies
    
    def evaluate_semantic_coherence(self, threshold=0.4, min_products=10):
        """Analyse sémantique : évalue la cohérence et sauvegarde les catégories problématiques et performantes"""
        print("\n" + "="*60)
        print("3 ÉVALUATION DE LA COHÉRENCE SÉMANTIQUE")
        print("="*60)
        
        if not EMBEDDINGS_AVAILABLE:
            print("\nSentence-transformers non disponible.")
            return None
        
        print("\nChargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Charger les noms de catégories
        category_names = self.load_category_names()
        
        # Analyser toutes les catégories avec suffisamment de produits
        category_counts = self.df['category_id'].value_counts()
        valid_categories = category_counts[category_counts >= min_products].index
        
        print(f"\nAnalyse de {len(valid_categories)} catégories...")
        
        low_coherence_data = []
        high_coherence_data = []
        texts_combined = (self.df['title'].fillna('') + ' ' + self.df['description'].fillna('')).str.strip()
        
        for cat_id in valid_categories:
            cat_products = self.df[self.df['category_id'] == cat_id]
            if len(cat_products) > 100:
                cat_products = cat_products.sample(n=100, random_state=42)
            
            texts = texts_combined[cat_products.index].tolist()
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            
            # Distance moyenne intra-classe
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings)
            np.fill_diagonal(distances, np.nan)
            avg_distance = np.nanmean(distances)
            coherence_score = 1 - avg_distance
            
            # Récupérer quelques exemples de titres
            example_titles = [
                str(title)[:80] for title in cat_products['title'].head(5).fillna('')
                if str(title).strip()
            ]
            
            category_info = {
                'category_id': cat_id,
                'category_name': category_names.get(cat_id, {}).get('name', 'N/A') if isinstance(category_names.get(cat_id), dict) else category_names.get(cat_id, 'N/A'),
                'category_path': cat_products.iloc[0]['category_path'],
                'n_products': len(cat_products),
                'coherence_score': round(float(coherence_score), 3),
                'example_titles': example_titles[:5]
            }
            
            if coherence_score < threshold:
                low_coherence_data.append(category_info)
            else:
                high_coherence_data.append(category_info)
        
        # Trier : low par score croissant, high par score décroissant
        low_coherence_data.sort(key=lambda x: x['coherence_score'])
        high_coherence_data.sort(key=lambda x: x['coherence_score'], reverse=True)
        
        # Sauvegarder low_coherence_categories.json
        output_path_low = Path(__file__).parent.parent / 'results' / 'audit' / 'low_coherence_categories.json'
        output_path_low.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_low, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_low_coherence_categories': len(low_coherence_data),
                'categories': low_coherence_data
            }, f, indent=2, ensure_ascii=False)
        
        # Sauvegarder high_coherence_categories.json
        output_path_high = Path(__file__).parent.parent / 'results' / 'audit' / 'high_coherence_categories.json'
        with open(output_path_high, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_high_coherence_categories': len(high_coherence_data),
                'categories': high_coherence_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 {len(low_coherence_data)} catégories à faible cohérence sauvegardées")
        if low_coherence_data:
            print(f"   Top 3: {', '.join([c['category_id'] for c in low_coherence_data[:3]])}")
        
        print(f"\n💾 {len(high_coherence_data)} catégories à haute cohérence sauvegardées")
        if high_coherence_data:
            print(f"   Top 3: {', '.join([c['category_id'] for c in high_coherence_data[:3]])}")
        
        return low_coherence_data, high_coherence_data
    
    def check_train_test_distribution(self, test_path):
        """Vérifie la cohérence des distributions train/test"""
        print("\n" + "="*60)
        print("4 VÉRIFICATION DISTRIBUTION TRAIN/TEST")
        print("="*60)
        
        df_test = pd.read_csv(test_path)
        
        # Calculer les proportions par catégorie
        train_props = self.df['category_id'].value_counts(normalize=True).sort_index()
        test_props = df_test['category_id'].value_counts(normalize=True).sort_index()
        
        # Vérifier les catégories manquantes
        missing_in_test = set(train_props.index) - set(test_props.index)
        missing_in_train = set(test_props.index) - set(train_props.index)
        
        if missing_in_train:
            print(f"\n⚠️  {len(missing_in_train)} catégories du test absentes du train")
        
        # Aligner les index pour la corrélation
        common_cats = sorted(set(train_props.index) & set(test_props.index))
        train_aligned = train_props[common_cats]
        test_aligned = test_props[common_cats]
        
        # Corrélation et différence moyenne
        correlation = np.corrcoef(train_aligned, test_aligned)[0, 1]
        avg_diff = np.mean(np.abs(train_aligned - test_aligned)) * 100
        
        print(f"\nDistribution Train vs Test:")
        print(f"   ✓ Corrélation: {correlation:.3f}")
        print(f"   ✓ Différence moyenne: {avg_diff:.2f} points")
        
        # Catégories avec écart > 2 points
        diffs = (train_aligned - test_aligned).abs() * 100
        large_diffs = diffs[diffs > 2]
        if len(large_diffs) > 0:
            print(f"   ⚠️  {len(large_diffs)} catégories avec écart > 2 points:")
            for cat_id, diff in large_diffs.head(5).items():
                print(f"      - {cat_id}: train {train_aligned[cat_id]*100:.1f}% vs test {test_aligned[cat_id]*100:.1f}% (diff: {diff:.1f})")
        else:
            print(f"   ✓ Toutes les catégories ont une distribution similaire")
    
    def generate_report(self, test_path=None):
        """Génère un rapport complet d'audit"""
        print("\n" + "="*60)
        print("RAPPORT D'AUDIT COMPLET")
        print("="*60)
        
        # Charger les données
        self.load_data()
        
        # Analyses
        self.analyze_structure()
        self.detect_inconsistencies()
        self.evaluate_semantic_coherence(threshold=0.4)
        
        # Vérification train/test si test_path fourni
        if test_path:
            test_full_path = Path(__file__).parent.parent / test_path if not Path(test_path).is_absolute() else Path(test_path)
            if test_full_path.exists():
                self.check_train_test_distribution(test_full_path)
            else:
                print(f"\n⚠️  Fichier test non trouvé: {test_full_path}")
        
        print("\n" + "="*60)
        print("✓ Audit terminé")
        print("="*60)


def main():
    """Point d'entrée principal"""
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    if not train_path.exists():
        print(f"Fichier non trouvé: {train_path}")
        return
    
    auditor = TaxonomyAuditor(train_path)
    auditor.generate_report(test_path='data/testset.csv' if test_path.exists() else None)


if __name__ == '__main__':
    main()

