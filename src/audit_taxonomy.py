"""
Step 1 - Audit of the existing taxonomy

This script analyzes the training dataset to:
1. Understand the taxonomy structure (depth, number of categories per level)
2. Detect structural inconsistencies in category_path
3. Evaluate semantic coherence of products within each category
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# For semantic embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Sentence-transformers not installed. Semantic analysis will be limited.")


class TaxonomyAuditor:
    """Taxonomy auditor for e-commerce products"""
    
    def __init__(self, data_path, sample_size=None):
        """
        Parameters:
        -----------
        data_path : str
            Path to the training CSV file
        sample_size : int, optional
            Number of samples for semantic analysis (None = all)
        """
        self.data_path = data_path
        self.sample_size = sample_size
        self.df = None
        self.embedding_model = None
        
    def load_data(self):
        """Load training data"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"   {len(self.df):,} products loaded")
        return self.df
    
    def analyze_structure(self):
        """Analyze taxonomy structure"""
        print("\n" + "="*60)
        print("1 TAXONOMY STRUCTURE ANALYSIS")
        print("="*60)
        
        # Path depth
        self.df['path_depth'] = self.df['category_path'].str.count('/') + 1
        depths = self.df['path_depth'].value_counts().sort_index()
        
        print(f"\nTaxonomy depth:")
        print(f"   Minimum depth: {depths.index.min()}")
        print(f"   Maximum depth: {depths.index.max()}")
        print(f"   Average depth: {self.df['path_depth'].mean():.2f}")
        print(f"   Median depth: {self.df['path_depth'].median():.0f}")
        
        print(f"\nDepth distribution:")
        for depth, count in depths.items():
            pct = (count / len(self.df)) * 100
            print(f"   Level {depth}: {count:,} products ({pct:.1f}%)")
        
        # Unique categories per level
        print(f"\nUnique categories per level:")
        max_depth = int(self.df['path_depth'].max())
        for level in range(1, max_depth + 1):
            categories_at_level = set()
            for path in self.df['category_path']:
                parts = path.split('/')
                if len(parts) >= level:
                    categories_at_level.add(parts[level - 1])
            print(f"   Level {level}: {len(categories_at_level):,} unique categories")
        
        # Leaf categories
        unique_leaf_categories = self.df['category_id'].nunique()
        print(f"\nLeaf categories (category_id):")
        print(f"   {unique_leaf_categories:,} unique leaf categories")
        print(f"   {len(self.df):,} total products")
        avg_products_per_category = len(self.df) / unique_leaf_categories
        print(f"   Average: {avg_products_per_category:.1f} products per category")
        
        # Products distribution per category
        category_counts = self.df['category_id'].value_counts()
        print(f"\nProducts distribution per category:")
        print(f"   Most frequent category: {category_counts.max():,} products")
        print(f"   Least frequent category: {category_counts.min():,} products")
        print(f"   Median: {category_counts.median():.0f} products")
        
        # Top 5 most frequent categories
        print(f"\nTop 5 most frequent categories:")
        top_5 = category_counts.head(5)
        for cat_id, count in top_5.items():
            print(f"   {cat_id}: {count:,} products")
        
        # Bottom 5 least frequent categories
        print(f"\nTop 5 least frequent categories:")
        bottom_5 = category_counts.tail(5)
        for cat_id, count in bottom_5.items():
            print(f"   {cat_id}: {count:,} products")
        
        # Generate category names
        category_names = self.generate_category_names()
        
        return {
            'depths': depths,
            'max_depth': max_depth,
            'unique_leaf_categories': unique_leaf_categories,
            'category_counts': category_counts,
            'category_names': category_names
        }
    
    def generate_category_names(self):
        """Generate simple names for each category based on frequent keywords"""
        print("\nGenerating category names...")
        
        # Simple stopwords (FR/DE/EN)
        stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'pour', 'avec', 'sans', 
                    'der', 'die', 'das', 'und', 'oder', 'für', 'mit', 'ohne',
                    'the', 'a', 'an', 'and', 'or', 'for', 'with', 'without',
                    'à', 'd', 'l', 'un', 'une', 'en', 'sur', 'par', 'dans'}
        
        category_data = {}
        
        for cat_id in self.df['category_id'].unique():
            cat_products = self.df[self.df['category_id'] == cat_id]
            titles = cat_products['title'].fillna('').astype(str).tolist()
            
            # Extract words (min 3 characters, alphanumeric)
            words = []
            for title in titles:
                title_words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', title.lower())
                words.extend([w for w in title_words if w not in stopwords])
            
            # Top 2-3 most frequent words
            if words:
                word_counts = Counter(words)
                top_words = [word for word, _ in word_counts.most_common(3)]
                category_name = ' '.join(top_words).title()
            else:
                category_name = "Unknown Category"
            
            # Example titles (max 5)
            example_titles = [t[:80] for t in titles[:5] if t.strip()]
            
            category_data[cat_id] = {
                'name': category_name,
                'example_titles': example_titles
            }
        
        # Save
        output_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(category_data, f, indent=2, ensure_ascii=False)
        
        print(f"   {len(category_data)} names generated")
        print(f"Saved to: {output_path}")
        
        return category_data
    
    def load_category_names(self):
        """Load category names from category_names.json"""
        names_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
        if names_path.exists():
            with open(names_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def detect_inconsistencies(self):
        """Detect structural inconsistencies in category_path"""
        print("\n" + "="*60)
        print("2 STRUCTURAL INCONSISTENCY DETECTION")
        print("="*60)
        
        inconsistencies = {
            'category_id_mismatch': [],
            'empty_paths': [],
            'invalid_paths': []
        }
        
        # Check category_id matches last path element
        print("\nVerifying category_id / category_path consistency...")
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
            print(f"{mismatches:,} inconsistencies detected (category_id != last path element)")
            print(f"   Examples (first 5):")
            for inc in inconsistencies['category_id_mismatch'][:5]:
                print(f"      - Product: {inc['product_id'][:8]}... | category_id: {inc['category_id']} | path end: {inc['last_path_id']}")
        else:
            print(f"   No inconsistencies detected")
        
        # Check empty or invalid paths
        print("\nChecking for empty or invalid paths...")
        empty_paths = self.df[self.df['category_path'].isna() | (self.df['category_path'] == '')]
        if len(empty_paths) > 0:
            print(f"{len(empty_paths):,} products with empty path")
            inconsistencies['empty_paths'] = empty_paths['product_id'].tolist()
        else:
            print(f"   No empty paths")
        
        # Check ID format in paths (hexadecimal expected)
        print("\nVerifying ID format in paths...")
        invalid_format = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            for part in path_parts:
                # Check if 8-character hexadecimal
                if len(part) != 8 or not all(c in '0123456789abcdef' for c in part.lower()):
                    invalid_format += 1
                    inconsistencies['invalid_paths'].append({
                        'product_id': row['product_id'],
                        'category_path': row['category_path'],
                        'invalid_part': part
                    })
                    break
        
        if invalid_format > 0:
            print(f"{invalid_format:,} paths with invalid ID format")
        else:
            print(f"   All IDs have valid format (8-character hex)")
        
        # Check consecutive duplicates in paths
        print("\nChecking for consecutive duplicates in paths...")
        consecutive_duplicates = 0
        for idx, row in self.df.iterrows():
            path_parts = row['category_path'].split('/')
            for i in range(len(path_parts) - 1):
                if path_parts[i] == path_parts[i + 1]:
                    consecutive_duplicates += 1
                    break
        
        if consecutive_duplicates > 0:
            print(f"{consecutive_duplicates:,} paths with consecutive duplicates")
        else:
            print(f"   No consecutive duplicates")
        
        return inconsistencies
    
    def evaluate_semantic_coherence(self, threshold=0.4, min_products=10):
        """Semantic analysis: evaluate coherence and save problematic/performant categories"""
        print("\n" + "="*60)
        print("3 SEMANTIC COHERENCE EVALUATION")
        print("="*60)
        
        if not EMBEDDINGS_AVAILABLE:
            print("\nSentence-transformers not available.")
            return None
        
        print("\nLoading embedding model...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Load category names
        category_names = self.load_category_names()
        
        # Analyze categories with enough products
        category_counts = self.df['category_id'].value_counts()
        valid_categories = category_counts[category_counts >= min_products].index
        
        print(f"\nAnalyzing {len(valid_categories)} categories...")
        
        low_coherence_data = []
        high_coherence_data = []
        texts_combined = (self.df['title'].fillna('') + ' ' + self.df['description'].fillna('')).str.strip()
        
        for cat_id in tqdm(valid_categories, desc="Semantic analysis"):
            cat_products = self.df[self.df['category_id'] == cat_id]
            if len(cat_products) > 100:
                cat_products = cat_products.sample(n=100, random_state=42)
            
            texts = texts_combined[cat_products.index].tolist()
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False, batch_size=128)
            
            # Average intra-class distance
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances(embeddings)
            np.fill_diagonal(distances, np.nan)
            avg_distance = np.nanmean(distances)
            coherence_score = 1 - avg_distance
            
            # Get example titles
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
        
        # Sort: low by ascending score, high by descending score
        low_coherence_data.sort(key=lambda x: x['coherence_score'])
        high_coherence_data.sort(key=lambda x: x['coherence_score'], reverse=True)
        
        # Save low_coherence_categories.json
        output_path_low = Path(__file__).parent.parent / 'results' / 'audit' / 'low_coherence_categories.json'
        output_path_low.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path_low, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_low_coherence_categories': len(low_coherence_data),
                'categories': low_coherence_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save high_coherence_categories.json
        output_path_high = Path(__file__).parent.parent / 'results' / 'audit' / 'high_coherence_categories.json'
        with open(output_path_high, 'w', encoding='utf-8') as f:
            json.dump({
                'threshold': threshold,
                'total_high_coherence_categories': len(high_coherence_data),
                'categories': high_coherence_data
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n{len(low_coherence_data)} low coherence categories saved")
        if low_coherence_data:
            print(f"   Top 3: {', '.join([c['category_id'] for c in low_coherence_data[:3]])}")
        
        print(f"\n{len(high_coherence_data)} high coherence categories saved")
        if high_coherence_data:
            print(f"   Top 3: {', '.join([c['category_id'] for c in high_coherence_data[:3]])}")
        
        return low_coherence_data, high_coherence_data
    
    def check_train_test_distribution(self, test_path):
        """Check train/test distribution consistency"""
        print("\n" + "="*60)
        print("4 TRAIN/TEST DISTRIBUTION CHECK")
        print("="*60)
        
        df_test = pd.read_csv(test_path)
        
        # Calculate proportions per category
        train_props = self.df['category_id'].value_counts(normalize=True).sort_index()
        test_props = df_test['category_id'].value_counts(normalize=True).sort_index()
        
        # Check missing categories
        missing_in_test = set(train_props.index) - set(test_props.index)
        missing_in_train = set(test_props.index) - set(train_props.index)
        
        if missing_in_train:
            print(f"\n{len(missing_in_train)} test categories missing from train")
        
        # Align indexes for correlation
        common_cats = sorted(set(train_props.index) & set(test_props.index))
        train_aligned = train_props[common_cats]
        test_aligned = test_props[common_cats]
        
        # Correlation and average difference
        correlation = np.corrcoef(train_aligned, test_aligned)[0, 1]
        avg_diff = np.mean(np.abs(train_aligned - test_aligned)) * 100
        
        print(f"\nTrain vs Test Distribution:")
        print(f"   Correlation: {correlation:.3f}")
        print(f"   Average difference: {avg_diff:.2f} points")
        
        # Categories with gap > 2 points
        diffs = (train_aligned - test_aligned).abs() * 100
        large_diffs = diffs[diffs > 2]
        if len(large_diffs) > 0:
            print(f"   {len(large_diffs)} categories with gap > 2 points:")
            for cat_id, diff in large_diffs.head(5).items():
                print(f"      - {cat_id}: train {train_aligned[cat_id]*100:.1f}% vs test {test_aligned[cat_id]*100:.1f}% (diff: {diff:.1f})")
        else:
            print(f"   All categories have similar distribution")
    
    def generate_report(self, test_path=None):
        """Generate complete audit report"""
        print("\n" + "="*60)
        print("COMPLETE AUDIT REPORT")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Analyses
        self.analyze_structure()
        self.detect_inconsistencies()
        self.evaluate_semantic_coherence(threshold=0.4)
        
        # Train/test check if test_path provided
        if test_path:
            test_full_path = Path(__file__).parent.parent / test_path if not Path(test_path).is_absolute() else Path(test_path)
            if test_full_path.exists():
                self.check_train_test_distribution(test_full_path)
            else:
                print(f"\nTest file not found: {test_full_path}")
        
        print("\n" + "="*60)
        print("Audit completed")
        print("="*60)


def main():
    """Main entry point"""
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'data' / 'trainset.csv'
    test_path = base_path / 'data' / 'testset.csv'
    
    if not train_path.exists():
        print(f"File not found: {train_path}")
        return
    
    auditor = TaxonomyAuditor(train_path)
    auditor.generate_report(test_path='data/testset.csv' if test_path.exists() else None)


if __name__ == '__main__':
    main()
