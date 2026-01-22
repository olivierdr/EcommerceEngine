# Classification E-commerce

## Objectif

Classification automatique de produits e-commerce dans des catégories feuilles à partir des informations disponibles (title, description, brand, color). Le projet commence par un audit complet de la taxonomie existante pour comprendre sa structure et identifier d'éventuelles incohérences, puis explore une approche de classification flat (baseline) avec identification des produits incertains pour validation humaine.

**Pour une compréhension approfondie des choix stratégiques, résultats détaillés et analyses, consultez le fichier [SYNTHESE.md](SYNTHESE.md).**

## Structure du Projet

```
ClassificationEcommerce/
├── data/                          # Données d'entrée (CSV)
│   ├── .gitkeep
│   ├── trainset.csv              # Données d'entraînement (30,520 produits)
│   └── testset.csv               # Données de test (7,631 produits)
│
├── src/                           # Code source
│   ├── audit_taxonomy.py         # Étape 1 : Audit de la taxonomie
│   └── classify_flat.py          # Classification flat (baseline)
│
├── results/                       # Résultats générés
│   ├── audit/                    # Résultats de l'audit
│   │   ├── category_names.json
│   │   ├── low_coherence_categories.json
│   │   └── high_coherence_categories.json
│   └── classification/           # Résultats de la classification
│       ├── flat_model.pkl
│       ├── certain_categories_analysis.json
│       ├── uncertain_categories_analysis.json
│       └── confusion_patterns.json
│
├── .cache/                        # Cache (embeddings, etc.)
├── requirements.txt
├── README.md                      # Ce fichier
└── SYNTHESE.md                    # Synthèse détaillée des approches et résultats
```

## Installation

1. **Créer un environnement virtuel** (recommandé) :
```bash
cd ./ClassificationEcommerce/
python3 -m venv venv
source venv/bin/activate 
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

**Important :** Placez vos fichiers `trainset.csv` et `testset.csv` dans le dossier `data/` avant d'exécuter les scripts.

## Approche Méthodologique

### Étape 1 : Audit de la Taxonomie

Avant toute classification, une analyse approfondie du jeu de données permet de :
- Comprendre la structure hiérarchique (profondeur, nombre de catégories par niveau)
- Détecter les incohérences structurelles dans les `category_path`
- Évaluer la cohérence sémantique des produits au sein de chaque catégorie

**Exécution :**
```bash
python3 src/audit_taxonomy.py
```

**Résultats :**
- 100 catégories feuilles identifiées
- Profondeur variable : 3 à 8 niveaux (médiane : 6)
- Aucune incohérence structurelle détectée
- 32 catégories avec faible cohérence sémantique (< 0.4)
- 68 catégories avec haute cohérence sémantique (≥ 0.4)
- Génération de noms de catégories à partir des mots-clés fréquents

**Fichiers générés (dans `results/audit/`) :**
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
python3 src/classify_flat.py
```

**Résultats :**
- Accuracy : 77.47% sur le test set
- 75.5% des produits avec confiance ≥ 0.5 (certains)
- 24.5% des produits avec confiance < 0.5 (incertains, nécessitent validation humaine)

**Fichiers générés (dans `results/classification/`) :**
- `flat_model.pkl` : Modèle entraîné sauvegardé
- `certain_categories_analysis.json` : Top 10 catégories avec produits certains
- `uncertain_categories_analysis.json` : Top 10 catégories problématiques
- `confusion_patterns.json` : Top 10 patterns de confusion entre catégories

## Technologies Utilisées

- **Python 3**
- **scikit-learn** : Classification (Logistic Regression)
- **sentence-transformers** : Embeddings multilingues (FR/DE/EN)
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques

## Documentation Complémentaire

Pour une analyse détaillée des choix stratégiques, des résultats complets, des exemples d'erreurs et des axes d'amélioration, consultez le fichier **[SYNTHESE.md](SYNTHESE.md)**.

