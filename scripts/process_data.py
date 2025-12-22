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

import torch
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.arabic_cleaner import apply_arabic_cleaning
from src.preprocessing.tokenizer import tokenize_arabic
from src.preprocessing.padding import padding_

def load_config(config_path="configs/default_config.yaml"):
    """
    Load YAML configuration
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "default_config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

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

    if raw_file.endswith(".csv"):
        df = pd.read_csv(raw_file)
    elif raw_file.endswith((".xls", ".xlsx")):
        df = pd.read_excel(raw_file)
    else:
        raise ValueError("File must be CSV or Excel")

    # TAKE ONLY 5000 LINES
    df = df.iloc[:5000].copy()
    print(f"✓ Loaded {len(df)} raw samples (limited to 5000)")

    # Ensure required columns
    if 'tweet' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'tweet' and 'label' columns")

    # 2️⃣ Apply Arabic cleaning
    df_clean = apply_arabic_cleaning(df.copy(), text_col='tweet')
    print("✓ Applied Arabic cleaning")

    # 3️⃣ Split into train/test
    train_df, test_df = train_test_split(
        df_clean,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=df_clean['label']
    )
    print(f"✓ Split into {len(train_df)} train and {len(test_df)} test samples")

    # Save raw cleaned train/test CSVs
    train_csv_path = os.path.join(processed_dir, "train.csv")
    test_csv_path  = os.path.join(processed_dir, "test.csv")
    # Clear files first
    open(train_csv_path, 'w', encoding='utf-8').close()
    open(test_csv_path, 'w', encoding='utf-8').close()

    train_df.to_csv(train_csv_path, index=False, encoding="utf-8")
    test_df.to_csv(test_csv_path, index=False, encoding="utf-8")
    print(f"✓ Saved raw train CSV: {train_csv_path}")
    print(f"✓ Saved raw test CSV: {test_csv_path}")

    # 4️⃣ Tokenize sequences
    final_list_train, encoded_train, final_list_test, encoded_test, vocab = tokenize_arabic(
        train_df['text'].values, train_df['label'].values,
        test_df['text'].values, test_df['label'].values,
        vocab_size=vocab_size
    )
    print(f"✓ Tokenized sequences. Vocabulary size: {len(vocab)}")
    print("Most common words:", list(vocab.keys())[:10])

    # 5️⃣ Pad sequences
    x_train_pad = padding_(final_list_train, seq_len=seq_len)
    x_test_pad  = padding_(final_list_test, seq_len=seq_len)

    # Save vocabulary
    with open(os.path.join(processed_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=4)


    print(f"✓ Saved processed tokenized data to {processed_dir}")

    # 7️⃣ Save separate .npy files for compatibility with train_embeddings.py

    # Convert lists of sequences to object arrays before saving

    # Clear existing files first
    open(os.path.join(processed_dir, "final_list_train.npy"), 'wb').close()
    open(os.path.join(processed_dir, "encoded_train.npy"), 'wb').close()
    open(os.path.join(processed_dir, "final_list_test.npy"), 'wb').close()
    open(os.path.join(processed_dir, "encoded_test.npy"), 'wb').close()

    np.save(os.path.join(processed_dir, "final_list_train.npy"), np.array(final_list_train, dtype=object))
    np.save(os.path.join(processed_dir, "encoded_train.npy"), np.array(encoded_train, dtype=object))
    np.save(os.path.join(processed_dir, "final_list_test.npy"), np.array(final_list_test, dtype=object))
    np.save(os.path.join(processed_dir, "encoded_test.npy"), np.array(encoded_test, dtype=object))
    print("✓ Saved separate .npy files for training embeddings")

if __name__ == "__main__":
    config = load_config()
    preprocess_and_save(config)
