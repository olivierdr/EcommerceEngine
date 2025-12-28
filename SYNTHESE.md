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

## Limites de l'approche

### Limites techniques

1. **Modèle simple** : Logistic Regression linéaire, incapable de capturer des interactions complexes entre features. Un modèle plus sophistiqué (BERT fine-tuné, Transformers) pourrait améliorer les performances.

2. **Features limitées** : 
   - Seulement title + description exploités
   - Brand et color non utilisés (pour éviter explosion dimensionnelle avec one-hot encoding)
   - Pas d'exploitation de la structure hiérarchique dans les embeddings

3. **Sur-apprentissage modéré** : Écart de 7.72 points entre train et test, suggérant que le modèle mémorise partiellement les patterns d'entraînement.

4. **Seuil de confiance fixe** : Le seuil à 0.5 est arbitraire et non adaptatif selon la catégorie. Certaines catégories nécessiteraient un seuil plus strict.

### Limites méthodologiques

1. **Pas d'exploitation de la hiérarchie** : L'approche flat ignore la structure hiérarchique, alors qu'une approche top-down pourrait réduire l'espace de décision et améliorer la cohérence métier.

2. **Pas de réentraînement** : Les corrections humaines sur les produits incertains ne sont pas intégrées automatiquement pour améliorer le modèle.

3. **Catégories problématiques non traitées** : Les 32 catégories à faible cohérence sémantique identifiées dans l'audit ne bénéficient pas d'un traitement spécifique (ex: règles métier, validation obligatoire).

4. **Pas de fallback hiérarchique** : En cas de faible confiance, le modèle ne propose pas de catégorie parente comme alternative, ce qui pourrait améliorer l'expérience utilisateur.

### Limites opérationnelles

1. **Pas de détection de nouvelles catégories** : Le modèle ne peut prédire que parmi les 100 catégories vues à l'entraînement. Un nouveau type de produit nécessiterait un réentraînement.

2. **Pas de monitoring en temps réel** : Aucun mécanisme pour détecter une dérive de performance ou des changements dans la distribution des produits.

3. **Validation humaine non intégrée** : Bien que les produits incertains soient identifiés, il n'y a pas de workflow automatisé pour intégrer les corrections humaines.

## Mise en perspective avec les pratiques e-commerce

Les grandes plateformes e-commerce combinent généralement plusieurs approches :
- un modèle de classification principal (souvent deep learning, de type BERT ou équivalent),
- une validation hiérarchique pour garantir la cohérence des prédictions,
- des règles métier et des mécanismes de fallback vers des catégories parentes,
- et, pour les cas ambigus, une intervention humaine via des interfaces de validation.

**Approche hybride envisagée (non implémentée) :**

Une approche hybride pourrait combiner la classification flat avec une validation hiérarchique : pour les produits à faible confiance, vérifier si les top-K prédictions partagent un parent commun et choisir parmi celles-ci. Cette stratégie permettrait d'exploiter la structure hiérarchique sans la complexité d'un modèle top-down complet. Cependant, pour un test technique, l'approche flat reste suffisante et démontre les concepts clés.

## Perspectives et questions ouvertes

Une fois une taxonomie jugée satisfaisante, plusieurs questions structurantes demeurent :

- **Intégration de nouveaux produits** : Comment intégrer automatiquement un nouveau produit dans la taxonomie, et à partir de quel seuil de confiance déclencher une validation humaine ?

- **Création de catégories** : Comment décider de la création ou du découpage de catégories (logique métier vs détection algorithmique) ? Dans quelle mesure la création de nouvelles catégories doit rester manuelle ou peut être partiellement automatisée ?

- **Amélioration continue** : Comment intégrer les corrections humaines pour améliorer progressivement le modèle ? Faut-il réentraîner périodiquement ou mettre en place un apprentissage actif ?

- **Monitoring et qualité** : Comment surveiller la qualité du classement en production ? Quels métriques suivre (accuracy globale, par catégorie, taux de validation humaine) ?

Ces éléments conditionnent la robustesse, la scalabilité et la maintenabilité du système de classification à long terme.

## Axes d'amélioration prioritaires

### Court terme

1. **Exploitation de brand/color** : Normaliser et encoder ces features de manière plus efficace (fréquence, embedding, plutôt que one-hot) pour enrichir les features sans explosion dimensionnelle.

2. **Seuil adaptatif** : Ajuster le seuil de confiance par catégorie selon leur difficulté intrinsèque (catégories à faible cohérence sémantique nécessitent un seuil plus strict).

3. **Fallback hiérarchique** : Pour les produits à très faible confiance, proposer la catégorie parente comme alternative plutôt qu'une feuille incorrecte.

### Moyen terme

1. **Modèles plus sophistiqués** : BERT fine-tuné sur le domaine e-commerce, ou Transformers adaptés à la classification hiérarchique.

2. **Apprentissage actif** : Réentraînement périodique avec les produits corrigés manuellement, en privilégiant les catégories problématiques.

3. **Features hiérarchiques** : Embeddings qui capturent explicitement la structure hiérarchique (position dans l'arbre, chemin complet).

### Long terme

1. **Système de feedback** : Pipeline automatisé pour intégrer les corrections humaines et améliorer continuellement le modèle.

2. **Détection de nouvelles catégories** : Mécanisme pour identifier automatiquement des produits non classables dans les catégories existantes.

3. **Optimisation continue** : A/B testing de différents modèles, monitoring de la performance en temps réel, alertes automatiques en cas de dérive.
