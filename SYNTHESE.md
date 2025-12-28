# Synthèse du projet – Classification de produits e-commerce

L'objectif de ce projet est d'améliorer le classement des produits au sein d'une taxonomie e-commerce, afin de garantir une meilleure visibilité des produits et une expérience utilisateur cohérente.

Avant de proposer un modèle de classification, il est essentiel de comprendre comment la taxonomie existante a été construite, d'en évaluer la qualité et d'identifier d'éventuelles incohérences.

Dans un contexte réel, plusieurs questions doivent être posées en amont : par qui la taxonomie a-t-elle été définie, selon quelle logique métier, et dans quelle mesure le processus est-il automatisé ou validé manuellement ?

Un point notable du jeu de données est l'absence de libellés explicites pour les catégories : celles-ci sont représentées uniquement par des identifiants, ce qui limite l'interprétabilité et impose de s'appuyer principalement sur la structure hiérarchique et le contenu des produits.

Du point de vue métier, les erreurs de classification peuvent être regroupées en deux grandes catégories :
- des produits importants ou à fort potentiel classés dans des catégories de niche, les rendant difficilement trouvables ;
- des produits mal classés mais restant accessibles via la recherche ou la navigation, avec un impact moindre sur l'expérience utilisateur.

Dans un environnement de production, ces analyses pourraient être enrichies par des signaux implicites tels que les taux de clic, les ajouts au panier ou l'absence totale d'interactions, qui peuvent indiquer des problèmes de classement.

## Démarche retenue

Le projet est structuré en deux grandes étapes complémentaires.

### 1. Audit de la taxonomie existante

Avant tout entraînement de modèle, un audit est réalisé sur le jeu de données d'entraînement afin de :
- analyser la structure réelle de la taxonomie (profondeur des chemins, nombre de catégories par niveau),
- détecter des incohérences structurelles dans les `category_path`,
- évaluer la cohérence sémantique des produits au sein de chaque catégorie à l'aide d'embeddings construits à partir des titres et descriptions.

Cette étape permet d'identifier des catégories ou des produits potentiellement bruités et d'améliorer la qualité des données utilisées pour l'apprentissage.

**Résultats clés de l'audit :**

- **Structure hiérarchique** : 100 catégories feuilles, profondeur variable de 3 à 8 niveaux (médiane : 6 niveaux), distribution équilibrée (~305 produits par catégorie en moyenne)
- **Cohérence structurelle** : Aucune incohérence détectée (category_id cohérent avec category_path, pas de paths vides ou invalides)
- **Cohérence sémantique** : 32 catégories présentent une faible cohérence sémantique (< 0.4), notamment des catégories génériques comme "Import Allemand Deluxe" (score : 0.193) ou "Ans Anglais Adibou" (score : 0.241). À l'inverse, 68 catégories montrent une haute cohérence (≥ 0.4), comme "Batterie Compatible Vhbw" (score : 0.655) ou "Vin Cave Bouteilles" (score : 0.636)
- **Génération de noms** : Extraction automatique de noms de catégories à partir des mots-clés fréquents dans les titres de produits, permettant une meilleure interprétabilité

### 2. Classification des catégories feuilles

Dans un second temps, un modèle de classification supervisée est développé pour prédire la catégorie feuille d'un produit à partir de ses informations textuelles.

**Approche retenue : Classification Flat (Baseline)**

Une approche "flat" est implémentée comme baseline : prédiction directe de la catégorie feuille parmi les 100 classes, sans exploitation explicite de la hiérarchie. Cette approche simple et efficace sert de référence pour évaluer la difficulté du problème.

**Architecture technique :**
- **Embeddings** : Modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2` pour encoder les titres et descriptions (384 dimensions)
- **Classifieur** : Logistic Regression (simple, rapide, interprétable)
- **Features** : Concaténation de title + description uniquement (brand et color non exploités pour éviter la sur-dimensionnalité)

**Résultats de performance :**

- **Accuracy globale** : 77.47% sur le test set (1,721 erreurs sur 7,631 produits)
- **Sur-apprentissage modéré** : Écart de 7.72 points entre train (85.19%) et test (77.47%), acceptable pour un modèle simple
- **Distribution de confiance** : 75.5% des produits avec confiance ≥ 0.5 (certains), 24.5% avec confiance < 0.5 (incertains)
- **Performance par niveau de confiance** : Les produits "certains" (confiance ≥ 0.5) présentent une accuracy estimée bien supérieure à ceux "incertains", validant l'utilité du score de confiance

**Analyses détaillées générées :**

- **Catégories certaines** : Top 10 catégories avec le plus de produits à haute confiance (ex: "Batterie Compatible Vhbw" avec 97 produits certains, avg_confidence : 0.93)
- **Catégories incertaines** : Top 10 catégories problématiques (ex: "Haut Parleur Noir" avec 62.1% d'incertitude, 59 produits incertains)
- **Patterns de confusion** : Top 10 paires de catégories régulièrement confondues (ex: "Machine Laver Linge" vs "Linge Lave Laver" : 24 cas, confusion_rate : 22.9%)

**Exemples d'erreurs typiques :**

1. **Confusion sémantique proche** : 
   - Catégorie vraie : "Machine Laver Linge" → Prédite : "Linge Lave Laver"
   - Produit : "Samsung WD91N642OOW Autonome Charge avant A Noir, Blanc machine à laver avec sèche linge"
   - **Analyse** : Catégories sémantiquement très proches, le modèle hésite entre deux formulations équivalentes

2. **Catégories génériques problématiques** :
   - Catégorie vraie : "Haut Parleur Noir" → Prédite : "Enceinte Bluetooth" (ou inversement)
   - Produit : "Barre de Son Portable Bluetooth 4.0 Enceintes sans Fil"
   - **Analyse** : Catégorie avec faible cohérence sémantique (score : 0.388), produits hétérogènes regroupés artificiellement

3. **Confusion entre catégories de jeux vidéo** :
   - Catégorie vraie : "Psp Collection Essentials" → Prédite : "Import Xbox Anglais"
   - Produit : "Street Fighter Ex 3"
   - **Analyse** : Confusion entre catégories de jeux vidéo différentes plateformes, probablement due à des titres de jeux similaires

## Axes d'amélioration

### Améliorations techniques

1. **Modèles plus sophistiqués** : Passer de Logistic Regression à des modèles plus puissants (BERT fine-tuné, Transformers) pour capturer des interactions complexes entre features et améliorer les performances.

2. **Enrichissement des features** : 
   - Intégrer brand et color de manière efficace (fréquence, embeddings plutôt que one-hot encoding)
   - Exploiter la structure hiérarchique dans les embeddings (position dans l'arbre, chemin complet)
   - Réduire le sur-apprentissage modéré (écart de 7.72 points entre train et test)

3. **Seuil de confiance adaptatif** : Remplacer le seuil fixe à 0.5 par un seuil adaptatif selon la catégorie, avec des seuils plus stricts pour les catégories problématiques.

### Améliorations méthodologiques

1. **Exploitation de la hiérarchie** : Implémenter une approche top-down ou hybride pour réduire l'espace de décision et améliorer la cohérence métier, tout en conservant la simplicité de l'approche flat.

2. **Intégration des corrections humaines** : Mettre en place un mécanisme de réentraînement automatique avec les produits corrigés manuellement, en privilégiant les catégories problématiques.

3. **Traitement spécifique des catégories problématiques** : Appliquer des règles métier ou une validation obligatoire pour les 32 catégories à faible cohérence sémantique identifiées dans l'audit.

4. **Fallback hiérarchique** : En cas de faible confiance, proposer une catégorie parente comme alternative pour améliorer l'expérience utilisateur.

### Améliorations opérationnelles

1. **Détection de nouvelles catégories** : Développer un mécanisme pour identifier automatiquement des produits non classables dans les catégories existantes et déclencher un réentraînement ou une création de catégorie.

2. **Monitoring en temps réel** : Mettre en place des mécanismes pour détecter une dérive de performance ou des changements dans la distribution des produits (alertes automatiques, dashboards).

3. **Workflow de validation humaine** : Automatiser l'intégration des corrections humaines dans un pipeline d'amélioration continue, avec apprentissage actif pour prioriser les produits à valider.

## Mise en perspective avec les pratiques e-commerce

Les grandes plateformes e-commerce combinent généralement plusieurs approches :
- un modèle de classification principal (souvent deep learning, de type BERT ou équivalent),
- une validation hiérarchique pour garantir la cohérence des prédictions,
- des règles métier et des mécanismes de fallback vers des catégories parentes,
- et, pour les cas ambigus, une intervention humaine via des interfaces de validation.

## Résumé et perspectives

L'approche retenue combine un **audit préalable** de la taxonomie et une **classification flat simple mais efficace** (77.47% d'accuracy), permettant d'identifier automatiquement les produits incertains pour validation humaine.

Les principaux **axes d'amélioration** identifiés concernent l'enrichissement du modèle (passage à BERT fine-tuné), l'exploitation efficace de brand/color et de la hiérarchie, ainsi que la mise en place de mécanismes de réentraînement et de monitoring pour une amélioration continue.
