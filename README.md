# Arabic NLP Project - File Structure
````
arabic_nlp_project/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation file (optional)
│
├── data/                             # Data directory
│   ├── raw/                          # Raw, unprocessed data
│   │   └── arabic_tweets.csv         # Your original dataset
│   ├── processed/                    # Cleaned and processed data
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── vocab.json                # Saved vocabulary
│   └── embeddings/                   # Saved embeddings
│       ├── word2vec_50d.npy          # Embedding matrix
│       └── embedding_metadata.json   # Embedding info
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── preprocessing/                # Text preprocessing modules
│   │   ├── __init__.py
│   │   ├── arabic_cleaner.py         # Arabic text cleaning functions
│   │   └── tokenizer.py              # Tokenization functions
│   │
│   ├── embeddings/                   # Word embedding modules
│   │   ├── __init__.py
│   │   ├── sgns_trainer.py           # SGNS/Word2Vec training
│   │   └── embedding_utils.py        # Embedding helper functions
│   │
│   ├── features/                     # Feature extraction modules
│   │   ├── __init__.py
│   │   ├── statistical_selector.py   # Chi2, MI selection
│   │   ├── dimensionality_reducer.py # PCA, SVD
│   │   ├── embedding_extractor.py    # Pooling methods
│   │   └── pipeline.py               # Complete feature pipeline
│   │
│   ├── models/                       # ML model modules
│   │   ├── __init__.py
│   │   ├── classifier.py             # Classification models
│   │   └── evaluation.py             # Evaluation metrics
│   │
│   └── utils/                        # Utility functions
│       ├── __init__.py
│       ├── data_loader.py            # Data loading utilities
│       ├── visualization.py          # Plotting functions
│       └── logger.py                 # Logging configuration
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # EDA and data analysis
│   ├── 02_preprocessing.ipynb        # Text cleaning experiments
│   ├── 03_embedding_training.ipynb   # Word2Vec training
│   ├── 04_feature_selection.ipynb    # Feature extraction experiments
│   ├── 05_model_training.ipynb       # Model training and tuning
│   └── 06_final_evaluation.ipynb     # Final results and comparison
│
├── scripts/                          # Standalone scripts
│   ├── train_embeddings.py           # Script to train embeddings
│   ├── extract_features.py           # Script to extract features
│   ├── train_classifier.py           # Script to train classifier
│   ├── predict.py                    # Script for predictions
│   └── run_pipeline.py               # End-to-end pipeline script
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_features.py
│   └── test_models.py
│
├── configs/                          # Configuration files
│   ├── default_config.yaml           # Default hyperparameters
│   ├── embedding_config.yaml         # Embedding training config
│   └── model_config.yaml             # Model training config
│
├── results/                          # Experimental results
│   ├── logs/                         # Training logs
│   ├── models/                       # Saved model files
│   │   ├── best_model.pkl
│   │   └── model_checkpoints/
│   ├── figures/                      # Plots and visualizations
│   └── reports/                      # Classification reports
│       └── final_report.txt
│
└── docs/                             # Additional documentation
    ├── api_reference.md              # API documentation
    ├── user_guide.md                 # User guide
    ├── feature_selection_guide.md    # Feature selection guide
    └── examples/                     # Usage examples
        ├── basic_usage.py
        ├── advanced_usage.py
        └── custom_pipeline.py

`````
# 🌙 Classification de Textes Arabes avec Word Embeddings

Un pipeline complet pour la classification de textes arabes utilisant des embeddings Word2Vec entraînés sur mesure et des méthodes avancées d'extraction de caractéristiques. Conçu pour les tâches de NLP arabe avec un focus sur la modularité, la reproductibilité et la performance.

## ✨ Fonctionnalités

- 🔤 **Prétraitement de textes arabes** : Nettoyage, normalisation et tokenisation spécialisés pour l'arabe
- 🧠 **Embeddings personnalisés** : Implémentation Skip-Gram with Negative Sampling (SGNS)
- 📊 **Méthodes multiples d'extraction de caractéristiques** :
  - Sélection statistique (Chi-carré, Information Mutuelle)
  - Pooling d'embeddings (moyenne, max, somme, pondéré TF-IDF)
  - Réduction de dimensionnalité (PCA, SVD)
- 🎯 **Pipeline bout-en-bout** : Du texte brut au classifieur entraîné
- 📈 **Évaluation complète** : Multiples métriques et outils de visualisation
- ⚙️ **Configurable** : Système de configuration basé sur YAML

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- Gestionnaire de paquets pip

### Configuration

1. **Cloner le dépôt**
git clone https://github.com/votreusername/arabic_nlp_project.git
cd arabic_nlp_project


2. **Créer un environnement virtuel**
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

3. **Installer les dépendances**
pip install -r requirements.txt

4. **Télécharger les données NLTK** (pour les stopwords)
python -c "import nltk; nltk.download('stopwords')"

5. **Installer le package en mode développement** (optionnel)
pip install -e .

## 📖 Utilisation

**[GUIDE_EXECUTION.md](GUIDE_EXECUTION.md)**






