# **6. Guide d’exécution du pipeline pour l’entraînement du modèle**

Pour entraîner le modèle, il est nécessaire de suivre les étapes ci-dessous dans l’ordre indiqué. Chaque script doit être exécuté étape par étape pour garantir le bon fonctionnement du pipeline.

### **1. Exploration et préparation des données**

1. `data_exploration.py`
    - Analyse exploratoire des données pour comprendre leur distribution et leurs caractéristiques principales.
2. `process_data.py`
    - Nettoyage, tokenisation et préparation des données brutes pour l’entraînement.
3. `tf_idf_loader.py`
    - Création et chargement des vecteurs TF-IDF des textes.

### **2. Extraction et entraînement des modèles**

1. `extract_features.py`
    - Génération des features à partir des textes (TF-IDF, embeddings, etc.).
2. `train_embeddings.py`
    - Entraînement ou chargement des vecteurs de mots (Word2Vec, GloVe…).
3. `train_simple_model.py`
    - Entraînement d’un modèle simple (ex. FeedForwardNN sur TF-IDF) pour test rapide.
4. `train_classifier.py`
    - Entraînement des modèles plus complexes (LSTM, GRU, CNN) sur embeddings ou TF-IDF.

Pour entraîner le modèle, il est nécessaire de suivre les étapes ci-dessous dans l’ordre indiqué. Chaque script doit être exécuté étape par étape pour garantir le bon fonctionnement du pipeline.

### **1.  préparation des données**

1. `process_data.py`
    - Nettoyage, tokenisation et préparation des données brutes pour l’entraînement.
2. `tf_idf_loader.py`
    - Création et chargement des vecteurs TF-IDF des textes.

### **2. Extraction et entraînement des modèles**

1. `extract_features.py`
    - Génération des features à partir des textes (TF-IDF, embeddings, etc.).
2. `train_embeddings.py`
    - Entraînement ou chargement des vecteurs de mots (Word2Vec, GloVe…).
3. `train_simple_model.py`
    - Entraînement d’un modèle simple (ex. FeedForwardNN sur TF-IDF) pour test rapide.
4. `train_classifier.py`
    - Entraînement des modèles plus complexes (LSTM, GRU, CNN) sur embeddings ou TF-IDF.

en terminal :

```bash
# 1. Exploration et préparation des données
python process_data.py          # Nettoyage, tokenisation et préparation des données
python tf_idf_loader.py         # Création et chargement des vecteurs TF-IDF

# 2. Extraction et entraînement des modèles
python train_embeddings.py      # Entraînement ou chargement des vecteurs de mots
python train_simple_model.py    # Entraînement d’un modèle simple (FeedForwardNN sur TF-IDF)
python train_classifier.py      # Entraînement des modèles complexes (LSTM, GRU, CNN)

```

 **Remarque :** Exécuter les scripts dans cet ordre garantit que toutes les étapes du pipeline pour entraîner le modèle sont respectées. 

# **7. Guide d’Exécution de  Prediction API :**

## 1. **Présentation**

Cette API est construite avec **FastAPI** pour permettre la prédiction automatique de textes en arabe. Elle supporte différents types de modèles :

- **LSTM**
- **GRU**
- **CNN**
- **Simple FeedForwardNN**

Le pipeline de prédiction inclut :

1. Nettoyage et tokenisation des textes en arabe.
2. Transformation des textes en séquences numériques à l’aide du vocabulaire existant.
3. Padding des séquences pour uniformiser leur longueur.
4. Chargement du meilleur modèle entraîné.
5. Prédiction des classes ou labels des textes.

## 2. **Structure des fichiers importants**

```
project_root/
│
├── app.py                  # Script FastAPI principal (celui fourni)
├── configs/
│   └── model_config.yaml   # Configuration des modèles (seq_len, hyperparamètres, etc.)
├── data/
│   └── processed/
│       └── vocab.json      # Vocabulaire utilisé pour la tokenisation
├── results/
│   └── models/
│       └── LSTM_best_model.pkl
│       └── GRU_best_model.pkl
│       └── CNN_best_model.pkl
│       └── Simple_FeedForwardNN_best_model.pkl
├── src/
│   ├── preprocessing/
│   │   ├── arabic_cleaner.py
│   │   ├── tokenizer.py
│   │   └── padding.py
│   └── models/
│       └── neural_networks.py

```

## 3. **Guide d’installation**

1. **Créer un environnement virtuel** (recommandé) :

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

```

1. **Installer les dépendances** :

```bash
pip install fastapi uvicorn torch pydantic pyyaml numpy

```

1. **S’assurer que les modèles et le vocabulaire existent** :
- `data/processed/vocab.json` doit contenir le vocabulaire.
- `results/models/` doit contenir les fichiers `_best_model.pkl`.

## 4. **Lancer l’API**

Depuis le dossier du projet :

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000

```

- L’API sera disponible à l’adresse : `http://127.0.0.1:8000`
- La documentation interactive Swagger sera disponible : `http://127.0.0.1:8000/docs`

## 5. **Guide d’utilisation de l’endpoint `/predict`**

**Méthode :** POST

**URL :** `/predict`

**Payload JSON attendu :**

```json
{
  "texts": [
    "هذا نص عربي للاختبار",
    "نص آخر للتجربة"
  ],
  "model_type": "LSTM"
}

```

**Paramètres :**

- `texts` : liste de chaînes de caractères en arabe.
- `model_type` : type de modèle à utiliser (`"LSTM"`, `"GRU"`, `"CNN"`, `"Simple_FeedForwardNN"`).

**Exemple avec `curl` :**

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
-H "Content-Type: application/json" \
-d '{"texts": ["هذا نص عربي للاختبار"], "model_type": "LSTM"}'

```

**Réponse JSON :**

```json
{
  "predictions": [1, 0]  # Liste des labels prédits pour chaque texte
}

```

## 6. **Description du pipeline interne**

1. **Chargement du vocabulaire** : `vocab.json`
2. **Prétraitement** :
    - Nettoyage avec `clean_arabic_text`
    - Tokenisation avec `tokenize_arabic`
    - Padding avec `padding_` pour avoir une séquence de longueur fixe
3. **Chargement du modèle** :
    - Chargement du meilleur modèle depuis `results/models/`
    - Si c’est un modèle PyTorch, le mode `eval()` est activé
4. **Prédiction** :
    - Si le modèle a une méthode `predict`, elle est utilisée
    - Sinon, les séquences sont converties en `torch.Tensor` et passées au modèle

## 7. **Remarques importantes**

- Le modèle et le vocabulaire doivent correspondre (le vocabulaire utilisé pour le tokenizing doit être le même que celui utilisé pour entraîner le modèle).
- Les séquences sont tronquées ou complétées à `seq_len` défini dans `model_config.yaml`.
- L’API gère les erreurs suivantes :
    - Texte vide ou incorrect → 400
    - Modèle non trouvé → 404
    - Problème interne de prédiction → 500
