"""
TF-IDF Loader
- Loads preprocessed tokenized data (.npy) and vocabulary
- Computes TF-IDF matrices
- Saves results as .npy files under processed directory
"""

import os
import json
import yaml
import numpy as np
from collections import Counter
from pathlib import Path

# ===============================
# Load config
# ===============================
def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# ===============================
# Load tokenized data
# ===============================
def load_tokenized_data(config: dict):
    processed_dir = Path(config["data"]["processed_dir"])

    x_train_file = processed_dir / "final_list_train.npy"
    y_train_file = processed_dir / "encoded_train.npy"
    x_test_file  = processed_dir / "final_list_test.npy"
    y_test_file  = processed_dir / "encoded_test.npy"
    vocab_file   = processed_dir / config["data"]["vocab_file"]

    # Check files
    for f in [x_train_file, y_train_file, x_test_file, y_test_file, vocab_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing file: {f}")

    # Load data
    x_train = np.load(x_train_file, allow_pickle=True)
    y_train = np.load(y_train_file, allow_pickle=True)
    x_test  = np.load(x_test_file, allow_pickle=True)
    y_test  = np.load(y_test_file, allow_pickle=True)

    # Load vocab
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return x_train, y_train, x_test, y_test, vocab

# ===============================
# Compute TF-IDF
# ===============================
def compute_tfidf(tokenized_train, tokenized_test, vocab_size: int):
    N = len(tokenized_train)

    # Document frequency
    doc_freq = np.zeros(vocab_size, dtype=np.int32)
    for doc in tokenized_train:
        if not doc:
            continue
        uniq = np.unique([i for i in doc if 1 <= i <= vocab_size])
        doc_freq[uniq - 1] += 1

    # IDF
    idf = np.log((N + 1) / (doc_freq + 1)) + 1.0
    idf = idf.astype(np.float32)

    def tfidf_matrix(docs):
        X = np.zeros((len(docs), vocab_size), dtype=np.float32)
        for r, doc in enumerate(docs):
            if not doc:
                continue
            counts = Counter(i for i in doc if 1 <= i <= vocab_size)
            length = len(doc)
            if length == 0:
                continue
            for idx1b, cnt in counts.items():
                X[r, idx1b - 1] = (cnt / length) * idf[idx1b - 1]
        return X

    X_train = tfidf_matrix(tokenized_train)
    X_test  = tfidf_matrix(tokenized_test)
    return X_train, X_test, idf

# ===============================
# Main
# ===============================
def main(config_path=None):
    config = load_config(config_path)

    print("✓ Loaded configuration")

    x_train, y_train, x_test, y_test, vocab = load_tokenized_data(config)
    print(f"✓ Loaded tokenized data. Vocabulary size: {len(vocab)}")

    vocab_size = len(vocab)
    X_train, X_test, idf = compute_tfidf(x_train, x_test, vocab_size=vocab_size)
    print("✓ Computed TF-IDF matrices")

    # Save TF-IDF matrices
    processed_dir = Path(config["data"]["processed_dir"])
    np.save(processed_dir / "X_train_tfidf.npy", X_train)
    np.save(processed_dir / "X_test_tfidf.npy", X_test)
    np.save(processed_dir / "idf_vector.npy", idf)
    np.save(processed_dir / "y_train.npy", y_train)
    np.save(processed_dir / "y_test.npy", y_test)

    print(f"✅ Saved TF-IDF features and labels to {processed_dir}")

if __name__ == "__main__":
    main()
