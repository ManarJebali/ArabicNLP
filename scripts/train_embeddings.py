"""
Train Word2Vec embeddings from processed data
Usage:
    python scripts/train_embeddings.py
"""

import json
import yaml
import numpy as np
from pathlib import Path

from src.embeddings.sgns_trainer import train_sgns_for_tokenize_outputs


# =========================================================
# Config loader
# =========================================================
def load_yaml_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================================================
# Data loader
# =========================================================
def load_tokenized_data():
    """
    Load tokenized .npy files and vocab.json
    """
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "processed"

    files = {
        "x_train": data_dir / "final_list_train.npy",
        "y_train": data_dir / "encoded_train.npy",
        "x_test":  data_dir / "final_list_test.npy",
        "y_test":  data_dir / "encoded_test.npy",
        "vocab":   data_dir / "vocab.json",
    }

    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    x_train = np.load(files["x_train"], allow_pickle=True)
    y_train = np.load(files["y_train"], allow_pickle=True)
    x_test  = np.load(files["x_test"], allow_pickle=True)
    y_test  = np.load(files["y_test"], allow_pickle=True)

    with open(files["vocab"], "r", encoding="utf-8") as f:
        vocab = json.load(f)

    return x_train, y_train, x_test, y_test, vocab


# =========================================================
# Main
# =========================================================
def main():
    project_root = Path(__file__).resolve().parents[1]

    # -----------------------------------------------------
    # Load embedding config
    # -----------------------------------------------------
    config_path = project_root / "configs" / "embedding_config.yaml"
    config = load_yaml_config(config_path)

    print("✓ Loaded embedding configuration")

    # -----------------------------------------------------
    # Load tokenized data
    # -----------------------------------------------------
    final_list_train, _, _, _, vocab = load_tokenized_data()

    print("✓ Loaded tokenized data")
    print(f"✓ Vocabulary size: {len(vocab)}")

    if not vocab:
        raise ValueError("Vocabulary is empty. Run preprocessing first.")

    # -----------------------------------------------------
    # Train embeddings
    # -----------------------------------------------------
    embedding_matrix, vocab_size = train_sgns_for_tokenize_outputs(
        final_list_train=final_list_train,
        onehot_dict=vocab,
        embed_dim=config["embedding_dim"],
        window_size=config["window_size"],
        num_negatives=config["negative_samples"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        lr=config["learning_rate"],
    )

    print(f"✓ Embedding matrix shape: {embedding_matrix.shape}")

    # -----------------------------------------------------
    # Save outputs USING CONFIG PATHS
    # -----------------------------------------------------
    embedding_path = project_root / config["embedding_output"]
    metadata_path  = project_root / config["metadata_output"]

    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    # Save embedding matrix
    np.save(embedding_path, embedding_matrix)

    # Save metadata
    metadata = {
        "embedding_method": config["embedding_method"],
        "tokenizer": config["tokenizer"],
        "embedding_dim": config["embedding_dim"],
        "window_size": config["window_size"],
        "negative_samples": config["negative_samples"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "learning_rate": config["learning_rate"],
        "vocab_size": vocab_size,
        "embedding_shape": list(embedding_matrix.shape),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"✅ Embeddings saved to: {embedding_path}")
    print(f"✅ Metadata saved to:   {metadata_path}")
    print("🎉 EMBEDDING TRAINING COMPLETE!")


# =========================================================
# Entry point
# =========================================================
if __name__ == "__main__":
    main()
