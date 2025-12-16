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
    print("⚠️  sentence-transformers non installé. L'analyse sémantique sera limitée.")


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
        print("📊 Chargement des données...")
        self.df = pd.read_csv(self.data_path)
        print(f"   ✓ {len(self.df):,} produits chargés")
        return self.df
    
    def analyze_structure(self):
        """Analyse la structure de la taxonomie"""
        print("\n" + "="*60)
        print("1️⃣  ANALYSE DE LA STRUCTURE DE LA TAXONOMIE")
        print("="*60)
        
        # Profondeur des chemins
        self.df['path_depth'] = self.df['category_path'].str.count('/') + 1
        depths = self.df['path_depth'].value_counts().sort_index()
        
        print(f"\n📏 Profondeur de la taxonomie:")
        print(f"   Profondeur minimale: {depths.index.min()}")
        print(f"   Profondeur maximale: {depths.index.max()}")
        print(f"   Profondeur moyenne: {self.df['path_depth'].mean():.2f}")
        print(f"   Profondeur médiane: {self.df['path_depth'].median():.0f}")
        
        print(f"\n📊 Distribution des profondeurs:")
        for depth, count in depths.items():
            pct = (count / len(self.df)) * 100
            print(f"   Niveau {depth}: {count:,} produits ({pct:.1f}%)")
        
        # Nombre de catégories uniques par niveau
        print(f"\n🏷️  Nombre de catégories uniques par niveau:")
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
        print(f"\n🍃 Catégories feuilles (category_id):")
        print(f"   {unique_leaf_categories:,} catégories feuilles uniques")
        print(f"   {len(self.df):,} produits au total")
        avg_products_per_category = len(self.df) / unique_leaf_categories
        print(f"   Moyenne: {avg_products_per_category:.1f} produits par catégorie")
        
        # Distribution des produits par catégorie
        category_counts = self.df['category_id'].value_counts()
        print(f"\n📈 Distribution des produits par catégorie:")
        print(f"   Catégorie la plus fréquente: {category_counts.max():,} produits")
        print(f"   Catégorie la moins fréquente: {category_counts.min():,} produits")
        print(f"   Médiane: {category_counts.median():.0f} produits")
        
        # Top 5 catégories les plus fréquentes
        print(f"\n🏆 Top 5 catégories les plus fréquentes:")
        top_5 = category_counts.head(5)
        for cat_id, count in top_5.items():
            cat_products = self.df[self.df['category_id'] == cat_id]
            example_title = cat_products.iloc[0]['title'] if len(cat_products) > 0 else "N/A"
            example_title = example_title[:60] + "..." if len(str(example_title)) > 60 else example_title
            print(f"   {cat_id}: {count:,} produits | Ex: {example_title}")
        
        # Bottom 5 catégories les moins fréquentes
        print(f"\n📉 Top 5 catégories les moins fréquentes:")
        bottom_5 = category_counts.tail(5)
        for cat_id, count in bottom_5.items():
            cat_products = self.df[self.df['category_id'] == cat_id]
            example_title = cat_products.iloc[0]['title'] if len(cat_products) > 0 else "N/A"
            example_title = example_title[:60] + "..." if len(str(example_title)) > 60 else example_title
            print(f"   {cat_id}: {count:,} produits | Ex: {example_title}")
        
        # Catégories avec peu de produits (potentiellement problématiques)
        rare_categories = (category_counts < 5).sum()
        print(f"\n⚠️  Catégories rares (< 5 produits): {rare_categories:,} ({rare_categories/unique_leaf_categories*100:.1f}%)")
        
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
        print("\n🏷️  Génération des noms de catégories...")
        
        # Stopwords simples (FR/DE/EN)
        stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'pour', 'avec', 'sans', 
                    'der', 'die', 'das', 'und', 'oder', 'für', 'mit', 'ohne',
                    'the', 'a', 'an', 'and', 'or', 'for', 'with', 'without',
                    'à', 'd', 'l', 'un', 'une', 'en', 'sur', 'par', 'dans'}
        
        category_names = {}
        
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
                category_names[cat_id] = category_name
            else:
                category_names[cat_id] = "Catégorie inconnue"
        
        # Sauvegarder
        output_path = Path(__file__).parent / 'category_names.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(category_names, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ {len(category_names)} noms générés")
        print(f"   💾 Sauvegardés dans: {output_path}")
        
        # Afficher quelques exemples
        print(f"\n   Exemples de noms générés:")
        for i, (cat_id, name) in enumerate(list(category_names.items())[:5]):
            print(f"   {cat_id}: {name}")
        
        return category_names
    
    def detect_inconsistencies(self):
        """Détecte les incohérences structurelles dans les category_path"""
        print("\n" + "="*60)
        print("2️⃣  DÉTECTION D'INCOHÉRENCES STRUCTURELLES")
        print("="*60)
        
        inconsistencies = {
            'category_id_mismatch': [],
            'empty_paths': [],
            'invalid_paths': []
        }
        
        # Vérifier que category_id correspond au dernier élément du path
        print("\n🔍 Vérification de cohérence category_id / category_path...")
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
            print(f"   ⚠️  {mismatches:,} incohérences détectées (category_id ≠ dernier élément du path)")
            print(f"   Exemples (premiers 5):")
            for inc in inconsistencies['category_id_mismatch'][:5]:
                print(f"      - Product: {inc['product_id'][:8]}... | category_id: {inc['category_id']} | path fin: {inc['last_path_id']}")
        else:
            print(f"   ✓ Aucune incohérence détectée")
        
        # Vérifier les paths vides ou invalides
        print("\n🔍 Vérification des paths vides ou invalides...")
        empty_paths = self.df[self.df['category_path'].isna() | (self.df['category_path'] == '')]
        if len(empty_paths) > 0:
            print(f"   ⚠️  {len(empty_paths):,} produits avec path vide")
            inconsistencies['empty_paths'] = empty_paths['product_id'].tolist()
        else:
            print(f"   ✓ Aucun path vide")
        
        # Vérifier les paths avec des IDs invalides (format hexadécimal attendu)
        print("\n🔍 Vérification du format des IDs dans les paths...")
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
            print(f"   ⚠️  {invalid_format:,} paths avec format d'ID invalide")
        else:
            print(f"   ✓ Tous les IDs ont un format valide (8 caractères hex)")
        
        # Vérifier les chemins avec des doublons consécutifs
        print("\n🔍 Vérification des doublons consécutifs dans les paths...")
        consecutive_duplicates = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            for i in range(len(path_parts) - 1):
                if path_parts[i] == path_parts[i + 1]:
                    consecutive_duplicates += 1
                    break
        
        if consecutive_duplicates > 0:
            print(f"   ⚠️  {consecutive_duplicates:,} paths avec doublons consécutifs")
        else:
            print(f"   ✓ Aucun doublon consécutif")
        
        return inconsistencies
    
    def evaluate_semantic_coherence(self, threshold=0.4, min_products=10):
        """Analyse sémantique simplifiée : évalue la cohérence et sauvegarde les catégories problématiques"""
        print("\n" + "="*60)
        print("3️⃣  ÉVALUATION DE LA COHÉRENCE SÉMANTIQUE")
        print("="*60)
        
        if not EMBEDDINGS_AVAILABLE:
            print("\n⚠️  sentence-transformers non disponible.")
            return None
        
        print("\n🔄 Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Analyser toutes les catégories avec suffisamment de produits
        category_counts = self.df['category_id'].value_counts()
        valid_categories = category_counts[category_counts >= min_products].index
        
        print(f"\n🔍 Analyse de {len(valid_categories)} catégories...")
        
        low_coherence_data = []
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
            
            if coherence_score < threshold:
                low_coherence_data.append({
                    'category_id': cat_id,
                    'category_path': cat_products.iloc[0]['category_path'],
                    'n_products': len(cat_products),
                    'coherence_score': float(coherence_score)
                })
        
        # Sauvegarder
        low_coherence_data.sort(key=lambda x: x['coherence_score'])
        output_path = Path(__file__).parent / 'low_coherence_categories.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_low_coherence_categories': len(low_coherence_data),
                'categories': low_coherence_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 {len(low_coherence_data)} catégories à faible cohérence sauvegardées")
        if low_coherence_data:
            print(f"   Top 3: {', '.join([c['category_id'] for c in low_coherence_data[:3]])}")
        
        return low_coherence_data
    
    def generate_report(self):
        """Génère un rapport complet d'audit"""
        print("\n" + "="*60)
        print("📋 RAPPORT D'AUDIT COMPLET")
        print("="*60)
        
        # Charger les données
        self.load_data()
        
        # Analyses
        structure_info = self.analyze_structure()
        inconsistencies = self.detect_inconsistencies()
        semantic_results = self.evaluate_semantic_coherence(threshold=0.4)
        
        # Résumé
        print("\n" + "="*60)
        print("📌 RÉSUMÉ")
        print("="*60)
        
        total_issues = (
            len(inconsistencies['category_id_mismatch']) +
            len(inconsistencies['empty_paths']) +
            len(inconsistencies['invalid_paths'])
        )
        
        print(f"\n✅ Points positifs:")
        print(f"   - {structure_info['unique_leaf_categories']:,} catégories feuilles identifiées")
        print(f"   - Profondeur maximale: {structure_info['max_depth']} niveaux")
        
        if total_issues > 0:
            print(f"\n⚠️  Points d'attention:")
            print(f"   - {total_issues:,} incohérences structurelles détectées")
            if len(inconsistencies['category_id_mismatch']) > 0:
                print(f"     • {len(inconsistencies['category_id_mismatch'])} mismatches category_id/path")
            if len(inconsistencies['empty_paths']) > 0:
                print(f"     • {len(inconsistencies['empty_paths'])} paths vides")
            if len(inconsistencies['invalid_paths']) > 0:
                print(f"     • {len(inconsistencies['invalid_paths'])} paths avec format invalide")
        else:
            print(f"\n✅ Aucune incohérence structurelle majeure détectée")
        
        if semantic_results:
            print(f"   - {len(semantic_results)} catégories avec faible cohérence sémantique (< 0.4)")
        
        print("\n" + "="*60)
        print("✓ Audit terminé")
        print("="*60)


def main():
    """Point d'entrée principal"""
    data_path = Path(__file__).parent / 'data' / 'trainset.csv'
    
    if not data_path.exists():
        print(f"❌ Fichier non trouvé: {data_path}")
        return
    
    auditor = TaxonomyAuditor(data_path)
    auditor.generate_report()


if __name__ == '__main__':
    main()

