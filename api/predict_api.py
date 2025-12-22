from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import numpy as np
import torch
import pickle
import yaml
import json

# Import preprocessing
from src.preprocessing.arabic_cleaner import clean_arabic_text
from src.preprocessing.tokenizer import tokenize_arabic
from src.preprocessing.padding import padding_

# ML imports
from src.models.neural_networks import LSTMClassifier, GRUClassifier, CNNClassifier

app = FastAPI(title="Arabic NLP Prediction API")

# ----------------------------
# Request schema
# ----------------------------
class PredictionRequest(BaseModel):
    texts: list  # List of raw Arabic text
    model_type: str  # "LSTM", "GRU", "CNN", or "Simple_FeedForwardNN"


# ----------------------------
# Load model config
# ----------------------------
def load_model_config(model_type: str):
    project_root = Path(__file__).resolve().parents[1]
    model_cfg_path = project_root / "configs" / "model_config.yaml"
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["model_type"] = model_type
    return config, project_root


# ----------------------------
# Load best model
# ----------------------------
def load_best_model(model_type: str):
    model_dir = Path("scripts/results/models")
    best_model_path = model_dir / f"{model_type}_best_model.pkl"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found: {best_model_path}")
    with open(best_model_path, "rb") as f:
        model = pickle.load(f)
    return model


# ----------------------------
# Preprocessing pipeline
# ----------------------------
def preprocess_texts(texts, vocab, seq_len=50):
    # 1️⃣ Clean texts
    cleaned_texts = [clean_arabic_text(t) for t in texts]

    # 2️⃣ Tokenize using existing vocabulary
    tokenized_texts = []
    for sent in cleaned_texts:
        tokens = [vocab[w] for w in sent.split() if w in vocab]
        tokenized_texts.append(tokens)

    # 3️⃣ Pad sequences
    padded = padding_(tokenized_texts, seq_len)
    return padded


# ----------------------------
# Prediction endpoint
# ----------------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.texts or not isinstance(request.texts, list):
        raise HTTPException(status_code=400, detail="`texts` must be a non-empty list of strings")

    # Load config
    config, _ = load_model_config(request.model_type)

    # Load vocab (required for tokenization)
    vocab_path = Path("data/processed/vocab.json")
    if not vocab_path.exists():
        raise HTTPException(status_code=500, detail="Vocabulary file not found")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Preprocess sequences
    seq_len = config.get("preprocessing", {}).get("seq_len", 50)
    sequences = preprocess_texts(request.texts, vocab, seq_len)

    # Load model
    try:
        model = load_best_model(request.model_type)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Make predictions
    try:
        if hasattr(model, "predict"):
            preds = model.predict(sequences)
        else:
            # PyTorch model
            model.eval()
            with torch.no_grad():
                x_tensor = torch.tensor(sequences, dtype=torch.long)
                output = model(x_tensor)
                preds = torch.argmax(output, dim=1).numpy()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return {"predictions": preds.tolist()}

