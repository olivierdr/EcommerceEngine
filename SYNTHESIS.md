# Project Synthesis – E-commerce Product Classification

The objective of this project is to improve product classification within an e-commerce taxonomy, to ensure better product visibility and a consistent user experience.

Before proposing a classification model, it is essential to understand how the existing taxonomy was constructed, evaluate its quality, and identify potential inconsistencies.

In a real-world context, several questions should be asked upfront: who defined the taxonomy, according to what business logic, and to what extent is the process automated or manually validated?

A notable aspect of the dataset is the absence of explicit labels for categories: they are represented only by identifiers, which limits interpretability and requires relying primarily on the hierarchical structure and product content.

From a business perspective, classification errors can be grouped into two main categories:
- important or high-potential products classified in niche categories, making them hard to find;
- misclassified products that remain accessible via search or navigation, with less impact on user experience.

In a production environment, these analyses could be enriched by implicit signals such as click rates, cart additions, or complete lack of interactions, which may indicate classification issues.

## Approach Taken

The project is structured in two main complementary steps.

### 1. Audit of the Existing Taxonomy

Before any model training, an audit is performed on the training dataset to:
- analyze the actual taxonomy structure (path depth, number of categories per level),
- detect structural inconsistencies in `category_path`,
- evaluate semantic coherence of products within each category using embeddings built from titles and descriptions.

This step helps identify potentially noisy categories or products and improve the quality of data used for training.

**Key Audit Results:**

- **Hierarchical structure**: 100 leaf categories, variable depth from 3 to 8 levels (median: 6 levels), balanced distribution (~305 products per category on average)
- **Structural coherence**: No inconsistencies detected (category_id consistent with category_path, no empty or invalid paths)
- **Semantic coherence**: 32 categories show low semantic coherence (< 0.4), notably generic categories like "Import Allemand Deluxe" (score: 0.193) or "Ans Anglais Adibou" (score: 0.241). Conversely, 68 categories show high coherence (>= 0.4), like "Batterie Compatible Vhbw" (score: 0.655) or "Vin Cave Bouteilles" (score: 0.636)
- **Name generation**: Automatic extraction of category names from frequent keywords in product titles, enabling better interpretability

### 2. Leaf Category Classification

In a second step, a supervised classification model is developed to predict a product's leaf category from its textual information.

**Approach Chosen: Flat Classification (Baseline)**

A "flat" approach is implemented as baseline: direct prediction of the leaf category among 100 classes, without explicit exploitation of the hierarchy. This simple and effective approach serves as reference to evaluate problem difficulty.

**Technical Architecture:**
- **Embeddings**: Multilingual model `paraphrase-multilingual-MiniLM-L12-v2` to encode titles and descriptions (384 dimensions)
- **Classifier**: Logistic Regression (simple, fast, interpretable)
- **Features**: Concatenation of title + description only (brand and color not used to avoid over-dimensionality)

**Performance Results:**

- **Overall accuracy**: 77.47% on test set (1,721 errors out of 7,631 products)
- **Moderate overfitting**: 7.72 point gap between train (85.19%) and test (77.47%), acceptable for a simple model
- **Confidence distribution**: 75.5% of products with confidence >= 0.5 (certain), 24.5% with confidence < 0.5 (uncertain)
- **Performance by confidence level**: "Certain" products (confidence >= 0.5) show estimated accuracy much higher than "uncertain" ones, validating the usefulness of confidence scores

**Detailed Analyses Generated:**

- **Certain categories**: Top 10 categories with most high-confidence products (e.g.: "Batterie Compatible Vhbw" with 97 certain products, avg_confidence: 0.93)
- **Uncertain categories**: Top 10 problematic categories (e.g.: "Haut Parleur Noir" with 62.1% uncertainty, 59 uncertain products)
- **Confusion patterns**: Top 10 regularly confused category pairs (e.g.: "Machine Laver Linge" vs "Linge Lave Laver": 24 cases, confusion_rate: 22.9%)

**Typical Error Examples:**

1. **Close semantic confusion**: 
   - True category: "Machine Laver Linge" → Predicted: "Linge Lave Laver"
   - Product: "Samsung WD91N642OOW Autonome Charge avant A Noir, Blanc machine à laver avec sèche linge"
   - **Analysis**: Semantically very close categories, model hesitates between two equivalent formulations

2. **Problematic generic categories**:
   - True category: "Haut Parleur Noir" → Predicted: "Enceinte Bluetooth" (or vice versa)
   - Product: "Barre de Son Portable Bluetooth 4.0 Enceintes sans Fil"
   - **Analysis**: Category with low semantic coherence (score: 0.388), heterogeneous products artificially grouped

3. **Confusion between video game categories**:
   - True category: "Psp Collection Essentials" → Predicted: "Import Xbox Anglais"
   - Product: "Street Fighter Ex 3"
   - **Analysis**: Confusion between video game categories for different platforms, probably due to similar game titles

## Improvement Axes

### Technical Improvements

1. **More sophisticated models**: Move from Logistic Regression to more powerful models (fine-tuned BERT, Transformers) to capture complex interactions between features and improve performance.

2. **Feature enrichment**: 
   - Efficiently integrate brand and color (frequency, embeddings rather than one-hot encoding)
   - Exploit hierarchical structure in embeddings (position in tree, full path)
   - Reduce moderate overfitting (7.72 point gap between train and test)

3. **Adaptive confidence threshold**: Replace fixed 0.5 threshold with adaptive threshold per category, with stricter thresholds for problematic categories.

### Methodological Improvements

1. **Hierarchy exploitation**: Implement a top-down or hybrid approach to reduce decision space and improve business coherence, while maintaining flat approach simplicity.

2. **Human corrections integration**: Set up automatic retraining mechanism with manually corrected products, prioritizing problematic categories.

3. **Specific handling of problematic categories**: Apply business rules or mandatory validation for the 32 low semantic coherence categories identified in the audit.

4. **Hierarchical fallback**: In case of low confidence, propose a parent category as alternative to improve user experience.

### Operational Improvements

1. **New category detection**: Develop mechanism to automatically identify products not classifiable in existing categories and trigger retraining or category creation.

2. **Real-time monitoring**: Set up mechanisms to detect performance drift or changes in product distribution (automatic alerts, dashboards).

3. **Human validation workflow**: Automate integration of human corrections into a continuous improvement pipeline, with active learning to prioritize products to validate.

## Context with E-commerce Practices

Major e-commerce platforms generally combine several approaches:
- a main classification model (often deep learning, BERT-type or equivalent),
- hierarchical validation to ensure prediction consistency,
- business rules and fallback mechanisms to parent categories,
- and, for ambiguous cases, human intervention via validation interfaces.

## Summary and Perspectives

The approach combines a **preliminary audit** of the taxonomy and **simple but effective flat classification** (77.47% accuracy), enabling automatic identification of uncertain products for human validation.

The main **improvement axes** identified concern model enrichment (move to fine-tuned BERT), efficient exploitation of brand/color and hierarchy, as well as setting up retraining and monitoring mechanisms for continuous improvement.

