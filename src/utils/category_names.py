"""
Category names generation and loading
"""
import json
import re
from pathlib import Path
from collections import Counter


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
    
    output_path = Path(__file__).parent.parent.parent / 'results' / 'audit' / 'category_names.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(category_data, f, indent=2, ensure_ascii=False)
    
    print(f"   {len(category_data)} names generated")
    return category_data


def load_category_names():
    """Load category names from category_names.json"""
    names_path = Path(__file__).parent.parent.parent / 'results' / 'audit' / 'category_names.json'
    if names_path.exists():
        with open(names_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(list(data.values())[0], dict):
                return {cat_id: data[cat_id]['name'] for cat_id in data}
            return data
    return {}

