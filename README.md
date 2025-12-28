# Classification E-commerce - Test Technique

## Objectif

Classification automatique de produits e-commerce dans des catégories feuilles à partir des informations disponibles (title, description, brand, color). Le projet explore une approche de classification flat (baseline) avec identification des produits incertains pour validation humaine.

**Pour une compréhension approfondie des choix stratégiques, résultats détaillés et analyses, consultez le fichier [SYNTHESE.md](SYNTHESE.md).**

## Structure du Projet

```
ClassificationEcommerce/
├── data/
│   ├── trainset.csv          # Données d'entraînement (30,520 produits)
│   └── testset.csv           # Données de test (7,631 produits)
├── audit_taxonomy.py         # Étape 1 : Audit de la taxonomie
├── classify_flat.py          # Classification flat (baseline)
├── requirements.txt          # Dépendances Python
├── README.md                 # Ce fichier
└── SYNTHESE.md               # Synthèse détaillée des approches et résultats
```

## Installation

```bash
pip install -r requirements.txt
```

## Approche Méthodologique

### Étape 1 : Audit de la Taxonomie

Avant toute classification, une analyse approfondie du jeu de données permet de :
- Comprendre la structure hiérarchique (profondeur, nombre de catégories par niveau)
- Détecter les incohérences structurelles dans les `category_path`
- Évaluer la cohérence sémantique des produits au sein de chaque catégorie

**Exécution :**
```bash
python3 audit_taxonomy.py
```

**Résultats :**
- 100 catégories feuilles identifiées
- Profondeur variable : 3 à 8 niveaux (médiane : 6)
- Aucune incohérence structurelle détectée
- 32 catégories avec faible cohérence sémantique (< 0.4)
- 68 catégories avec haute cohérence sémantique (≥ 0.4)
- Génération de noms de catégories à partir des mots-clés fréquents

**Fichiers générés :**
- `category_names.json` : Noms générés pour chaque catégorie avec exemples
- `low_coherence_categories.json` : Catégories à faible cohérence sémantique
- `high_coherence_categories.json` : Catégories à haute cohérence sémantique

### Étape 2 : Classification Flat (Baseline)

**Principe :** Prédiction directe de la catégorie feuille parmi les 100 classes, avec identification des produits incertains basée sur les scores de confiance.

**Architecture :**
- Embeddings : Modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)
- Classifieur : Logistic Regression
- Features : Concaténation de title + description

**Exécution :**
```bash
python3 classify_flat.py
```

**Résultats :**
- Accuracy : 77.47% sur le test set
- 75.5% des produits avec confiance ≥ 0.5 (certains)
- 24.5% des produits avec confiance < 0.5 (incertains, nécessitent validation humaine)

**Fichiers générés :**
- `flat_model.pkl` : Modèle entraîné sauvegardé
- `certain_categories_analysis.json` : Top 10 catégories avec produits certains
- `uncertain_categories_analysis.json` : Top 10 catégories problématiques
- `confusion_patterns.json` : Top 10 patterns de confusion entre catégories

## Résultats Principaux

### Performance Globale

- **Accuracy** : 77.47% sur le test set (1,721 erreurs sur 7,631 produits)
- **Sur-apprentissage modéré** : Écart de 7.72 points entre train (85.19%) et test (77.47%)
- **Distribution de confiance** : 75.5% produits certains, 24.5% produits incertains

### Insights Clés

1. **Performance solide** : 77.47% d'accuracy avec un modèle simple (Logistic Regression)
2. **Identification efficace** : 1,873 produits incertains identifiés (24.5% du test set) pour validation humaine
3. **Catégories problématiques** : Top 3 catégories avec le plus d'incertitude (ex: "Haut Parleur Noir" avec 62.1% d'incertitude)
4. **Patterns de confusion** : Identification des paires de catégories régulièrement confondues (ex: "Machine Laver Linge" vs "Linge Lave Laver")

## Technologies Utilisées

- **Python 3**
- **scikit-learn** : Classification (Logistic Regression)
- **sentence-transformers** : Embeddings multilingues (FR/DE/EN)
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques

## Documentation Complémentaire

Pour une analyse détaillée des choix stratégiques, des résultats complets, des exemples d'erreurs et des axes d'amélioration, consultez le fichier **[SYNTHESE.md](SYNTHESE.md)**.
