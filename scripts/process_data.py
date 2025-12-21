"""
Preprocessing pipeline for Arabic NLP project
- Loads raw data from data/raw/
- Cleans text
- Tokenizes and builds vocabulary
- Splits into train/test
- Pads sequences
- Saves processed data to data/processed/
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from src.preprocessing.arabic_cleaner import apply_arabic_cleaning
from src.preprocessing.tokenizer import tokenize_arabic
from src.preprocessing.padding import padding_

def load_config(config_path="configs/default_config.yaml"):
    """
    Load YAML configuration
    """
    # Resolve path relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "default_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Convert train_path to absolute path
    train_path = config["data"]["train_path"]
    if not os.path.isabs(train_path):
        config["data"]["train_path"] = os.path.join(project_root, train_path)

    return config

def preprocess_and_save(config):
    """
    Preprocess raw data and save processed files
    """
    raw_file = config["data"]["train_path"]
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/processed")
    os.makedirs(processed_dir, exist_ok=True)

    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    vocab_size = config["preprocessing"]["vocab_size"]
    seq_len = 50  # fixed sequence length for padding

    # 1️⃣ Load raw data (CSV or Excel)
    if raw_file.endswith(".csv"):
        df = pd.read_csv(raw_file)
    elif raw_file.endswith((".xls", ".xlsx")):
        df = pd.read_excel(raw_file)
    else:
        raise ValueError("File must be CSV or Excel")

    print(f"✓ Loaded {len(df)} raw samples")

    # Ensure required columns
    if 'tweet' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'tweet' and 'label' columns")

    # 2️⃣ Apply Arabic cleaning
    df_clean = apply_arabic_cleaning(df.copy(), text_col='tweet')
    print("✓ Applied Arabic cleaning")

    # 3️⃣ Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        df_clean['text'].values,
        df_clean['label'].values,
        test_size=test_size,
        random_state=random_state
    )
    print(f"✓ Split into {len(x_train)} train and {len(x_test)} test samples")

    # 4️⃣ Tokenize
    final_list_train, encoded_train, final_list_test, encoded_test, vocab = tokenize_arabic(
        x_train, y_train, x_test, y_test, vocab_size=vocab_size
    )
    print(f"✓ Tokenized sequences. Vocabulary size: {len(vocab)}")

    # Optional: show most common words
    print("Most common words:", list(vocab.keys())[:10])

    # 5️⃣ Pad sequences
    x_train_pad = padding_(final_list_train, seq_len=seq_len)
    x_test_pad  = padding_(final_list_test, seq_len=seq_len)

    # 6️⃣ Save processed data

    # Save padded arrays (safe for np.save)
    np.save(os.path.join(processed_dir, "x_train_pad.npy"), x_train_pad)
    np.save(os.path.join(processed_dir, "x_test_pad.npy"), x_test_pad)
    np.save(os.path.join(processed_dir, "encoded_train.npy"), encoded_train, allow_pickle=True)
    np.save(os.path.join(processed_dir, "encoded_test.npy"), encoded_test, allow_pickle=True)

    # Save raw token lists with pickle if needed
    with open(os.path.join(processed_dir, "final_list_train.pkl"), "wb") as f:
        pickle.dump(final_list_train, f)
    with open(os.path.join(processed_dir, "final_list_test.pkl"), "wb") as f:
        pickle.dump(final_list_test, f)

    # Save vocabulary
    with open(os.path.join(processed_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)

    print(f"✓ Saved processed data to {processed_dir}")
    return x_train_pad, encoded_train, x_test_pad, encoded_test, vocab

if __name__ == "__main__":
    # Load configuration
    config = load_config()

    # Run preprocessing
    preprocess_and_save(config)
