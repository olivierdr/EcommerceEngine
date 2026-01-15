"""
E-commerce product classifier training
"""

import pandas as pd
import numpy as np
import pickle
import json
import re
from pathlib import Path
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


class FlatClassifier:
    """Flat classifier for direct leaf category prediction"""
    
    def __init__(self, embedding_model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize embedding model"""
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        self.classifier = None
        self.label_encoder = None
        self.df_train = None
        self.X_train = None
        self.cat_to_path = {}
        
    def prepare_features(self, df, show_progress=True, cache_path=None):
        """Prepare text features (title + description)"""
        if cache_path and Path(cache_path).exists():
            if show_progress:
                print(f"Loading embeddings from cache...")
            return np.load(cache_path)
        
        if show_progress:
            print("Preparing features...")
        texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).str.strip()
        text_embeddings = self.embedding_model.encode(
            texts.tolist(), 
            show_progress_bar=show_progress,
            batch_size=128
        )
        
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, text_embeddings)
            if show_progress:
                print(f"   Cache saved: {cache_path}")
        
        if show_progress:
            print(f"   {len(texts):,} texts encoded to {text_embeddings.shape[1]}-dimensional vectors")
        return text_embeddings
    
    def train(self, train_path):
        """Train model on training set"""
        print("\n" + "="*60)
        print("TRAINING - Flat Classification")
        print("="*60)
        
        print("\nLoading training data...")
        df_train = pd.read_csv(train_path)
        print(f"   {len(df_train):,} products loaded")
        
        self.df_train = df_train
        self.X_train = self.prepare_features(df_train, show_progress=True)
        y_train = df_train['category_id'].values
        
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        print(f"\nStatistics:")
        print(f"   Number of categories: {len(self.label_encoder.classes_)}")
        print(f"   Embedding dimension: {self.X_train.shape[1]}")
        
        print("\nTraining classifier (Logistic Regression)...")
        self.classifier = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        self.classifier.fit(self.X_train, y_train_encoded)
        print("   Model trained")
        
        self.cat_to_path = dict(zip(df_train['category_id'], df_train['category_path']))
        
        return self
    
    def predict(self, df):
        """Predict categories for a dataframe"""
        X = self.prepare_features(df, show_progress=False)
        y_pred_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, df):
        """Return prediction probabilities"""
        X = self.prepare_features(df, show_progress=False)
        return self.classifier.predict_proba(X)
    
    def predict_with_confidence(self, df):
        """Predict with confidence scores"""
        X = self.prepare_features(df, show_progress=False)
        y_pred_encoded = self.classifier.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_pred_proba = self.classifier.predict_proba(X)
        return y_pred, np.max(y_pred_proba, axis=1)
    
    def save(self, model_path):
        """Save model"""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'label_encoder': self.label_encoder,
                'embedding_model_name': self.embedding_model_name,
                'cat_to_path': self.cat_to_path
            }, f)
        print(f"Model saved: {model_path}")
    
    @classmethod
    def load(cls, model_path):
        """Load a saved model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(embedding_model_name=data['embedding_model_name'])
        classifier.classifier = data['classifier']
        classifier.label_encoder = data['label_encoder']
        classifier.cat_to_path = data.get('cat_to_path', {})
        return classifier


def generate_category_names(df):
    """Generate simple names for each category based on frequent keywords"""
    print("\nGenerating category names...")
    
    stopwords = {'le', 'la', 'les', 'de', 'du', 'des', 'et', 'ou', 'pour', 'avec', 'sans', 
                'der', 'die', 'das', 'und', 'oder', 'für', 'mit', 'ohne',
                'the', 'a', 'an', 'and', 'or', 'for', 'with', 'without',
                'à', 'd', 'l', 'un', 'une', 'en', 'sur', 'par', 'dans'}
    
    category_data = {}
    
    for cat_id in df['category_id'].unique():
        cat_products = df[df['category_id'] == cat_id]
        titles = cat_products['title'].fillna('').astype(str).tolist()
        
        words = []
        for title in titles:
            title_words = re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', title.lower())
            words.extend([w for w in title_words if w not in stopwords])
        
        if words:
            word_counts = Counter(words)
            top_words = [word for word, _ in word_counts.most_common(3)]
            category_name = ' '.join(top_words).title()
        else:
            category_name = "Unknown Category"
        
        example_titles = [t[:80] for t in titles[:5] if t.strip()]
        
        category_data[cat_id] = {
            'name': category_name,
            'example_titles': example_titles
        }
    
    output_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(category_data, f, indent=2, ensure_ascii=False)
    
    print(f"   {len(category_data)} names generated")
    return category_data


def load_category_names():
    """Load category names from category_names.json"""
    names_path = Path(__file__).parent.parent / 'results' / 'audit' / 'category_names.json'
    if names_path.exists():
        with open(names_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(list(data.values())[0], dict):
                return {cat_id: data[cat_id]['name'] for cat_id in data}
            return data
    return {}


def main():
    print("\n" + "="*60)
    print("CLASSIFIER TRAINING")
    print("="*60)
    
    base_path = Path(__file__).parent.parent
    train_path = base_path / 'data' / 'trainset.csv'
    model_path = base_path / 'results' / 'classification' / 'flat_model.pkl'
    
    if not train_path.exists():
        print(f"File not found: {train_path}")
        return None
    
    # Train
    classifier = FlatClassifier()
    classifier.train(train_path)
    
    # Generate category names
    generate_category_names(classifier.df_train)
    
    # Save
    classifier.save(model_path)
    
    print("\n" + "="*60)
    print("Training completed")
    print("="*60)
    
    return classifier


if __name__ == '__main__':
    classifier = main()
