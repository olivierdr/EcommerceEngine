# Classification E-commerce - Test Technique

## 📋 Objectif

Classification automatique de produits e-commerce dans des catégories feuilles à partir des informations disponibles (title, description, brand, color). Le projet explore deux approches de classification et compare leurs performances.

## 🏗️ Structure du Projet

```
ClassificationEcommerce/
├── data/
│   ├── trainset.csv          # Données d'entraînement (30,520 produits)
│   └── testset.csv           # Données de test (7,631 produits)
├── audit_taxonomy.py         # Étape 1 : Audit de la taxonomie
├── classify_flat.py           # Approche 1 : Classification flat (baseline)
├── classify_hybrid.py         # Approche 2 : Classification hybride
├── compare_approaches.py      # Analyse comparative détaillée
├── requirements.txt            # Dépendances Python
└── README.md                  # Ce fichier
```

## 🚀 Installation

```bash
pip install -r requirements.txt
```

## 📊 Approche Méthodologique

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

### Étape 2 : Classification

#### Approche 1 : Flat Classification (Baseline)

**Principe :** Prédiction directe de la catégorie feuille parmi les 100 classes.

**Avantages :**
- Simplicité et rapidité
- Meilleure performance (accuracy : 77.47%)
- Un seul modèle à maintenir

**Exécution :**
```bash
python3 classify_flat.py
```

#### Approche 2 : Hybrid Classification

**Principe :** Classification flat avec identification des produits incertains et scores de confiance.

**Avantages :**
- Même performance que flat (77.47%)
- Identification des produits à faible confiance pour validation humaine
- Métriques de monitoring (1,873 produits incertains identifiés sur 7,631)
- Export JSON pour workflow de validation humaine

**Exécution :**
```bash
python3 classify_hybrid.py
```

**Sorties :**
- `uncertain_products/uncertain_products.json` : Produits avec confiance < 0.5

### Analyse Comparative

**Exécution :**
```bash
python3 compare_approaches.py
```

Génère un rapport complet (`comparison_report.json`) avec :
- Métriques comparatives détaillées
- Analyse des erreurs (top catégories confondues)
- Exemples d'erreurs
- Recommandations

## 📈 Résultats Principaux

### Performance Globale

| Approche | Accuracy | Temps Entraînement | Erreurs |
|----------|----------|-------------------|---------|
| **Flat** | 77.47% | ~30s | 1,721 (22.53%) |
| **Hybride** | 77.47% | ~35s | 1,721 (22.53%) |

### Insights Clés

1. **Performance identique** : Les deux approches atteignent la même accuracy (77.47%)
2. **Valeur ajoutée hybride** : Identification de 1,873 produits incertains (24.5% du test set)
3. **Accuracy produits haute confiance** : 88.61% sur les produits avec confiance ≥ 0.5
4. **Accuracy produits faible confiance** : 41.57% sur les produits avec confiance < 0.5

## 💡 Recommandations

### Pour la Production

**Approche recommandée : Hybride**

**Justification :**
- Performance identique à flat
- Identification automatique des produits nécessitant validation humaine
- Réduction de 75% du volume à valider manuellement (1,873 vs 7,631)
- Métriques de confiance pour monitoring qualité
- Workflow opérationnel prêt pour validation humaine

**Cas d'usage :**
- **Flat** : Production simple où seule la performance maximale compte
- **Hybride** : Production avec validation humaine, monitoring qualité, amélioration continue

## 🔍 Limitations et Améliorations Futures

### Limitations Actuelles

1. **Modèle simple** : Logistic Regression (pas de deep learning)
2. **Features basiques** : Seulement title + description (pas d'exploitation de brand/color)
3. **Pas de réentraînement** : Les corrections humaines ne sont pas intégrées automatiquement
4. **Seuil fixe** : Le seuil de confiance (0.5) n'est pas adaptatif

### Améliorations Possibles

1. **Modèles plus sophistiqués** : BERT fine-tuné, Transformers
2. **Features enrichies** : Exploitation de brand, color, embeddings hiérarchiques
3. **Apprentissage actif** : Réentraînement avec produits corrigés manuellement
4. **Seuil adaptatif** : Ajustement dynamique selon la catégorie
5. **Fallback hiérarchique** : Utiliser catégorie parente si confiance très faible

## 📝 Technologies Utilisées

- **Python 3**
- **scikit-learn** : Classification (Logistic Regression)
- **sentence-transformers** : Embeddings multilingues (FR/DE/EN)
- **pandas** : Manipulation de données
- **numpy** : Calculs numériques

## 📄 Fichiers Générés

- `rare_categories.json` : Catégories rares (< 10 produits)
- `low_coherence_categories.json` : Catégories à faible cohérence sémantique
- `uncertain_products/uncertain_products.json` : Produits incertains pour validation
- `comparison_report.json` : Rapport d'analyse comparative complet

## 👤 Auteur

Test technique - Classification E-commerce
