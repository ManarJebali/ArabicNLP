

import yaml
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.neural_networks import LSTMClassifier, GRUClassifier, CNNClassifier
from src.models.trainer import NeuralNetworkTrainer


# ----------------------------
# Load and merge YAML configs
# ----------------------------
def load_configs():
    project_root = Path(__file__).resolve().parents[1]

    default_cfg_path = project_root / "configs" / "default_config.yaml"
    embedding_cfg_path = project_root / "configs" / "embedding_config.yaml"
    model_cfg_path = project_root / "configs" / "model_config.yaml"

    with open(default_cfg_path, "r", encoding="utf-8") as f:
        default_cfg = yaml.safe_load(f)
    with open(embedding_cfg_path, "r", encoding="utf-8") as f:
        embedding_cfg = yaml.safe_load(f)
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Merge configs
    config = default_cfg.copy()
    config.update(embedding_cfg)
    config.update(model_cfg)
    return config, project_root


# ----------------------------
# Load preprocessed data & embeddings
# ----------------------------
def load_embeddings_and_data(config):
    """
    Load preprocessed data and embedding matrix
    """
    from pathlib import Path
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]

    # ----------------------------
    # Processed data
    # ----------------------------
    processed_dir = Path(config['data']['processed_dir'])
    if not processed_dir.is_absolute():
        processed_dir = project_root / processed_dir
    processed_dir = processed_dir.resolve()

    print(f"Looking for files in: {processed_dir}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    print("Available files:", list(processed_dir.iterdir()))

    # ----------------------------
    # Files
    # ----------------------------
    files = {
        "X_train": processed_dir / "X_train_tfidf.npy",
        "y_train": processed_dir / "y_train.npy",
        "X_test":  processed_dir / "X_test_tfidf.npy",
        "y_test":  processed_dir / "y_test.npy",
    }

    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # ----------------------------
    # Load numpy arrays
    # ----------------------------
    # Load numpy arrays
    X_train = np.load(files["X_train"], allow_pickle=True)
    y_train = np.load(files["y_train"], allow_pickle=True).astype(np.int64)  # <-- fix here
    X_test = np.load(files["X_test"], allow_pickle=True)
    y_test = np.load(files["y_test"], allow_pickle=True).astype(np.int64)  # <-- fix here

    # ----------------------------
    # Embedding matrix
    # ----------------------------
    embedding_path = Path(config['embeddings']['embedding_output'])
    if not embedding_path.is_absolute():
        embedding_path = project_root / embedding_path
    embedding_path = embedding_path.resolve()

    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")

    embedding_matrix = np.load(embedding_path, allow_pickle=True)

    return embedding_matrix, X_train, y_train, X_test, y_test

# ----------------------------
# Create DataLoaders
# ----------------------------
def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.long),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.long),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


# ----------------------------
# Model factory
# ----------------------------
def build_model(config, embedding_matrix):
    model_type = config["model_type"]
    input_dim = embedding_matrix.shape[0]  # vocab_size
    embedding_dim = config["model"]["embedding_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    num_layers = config["model"].get("num_layers", 1)
    bidirectional = config["model"].get("bidirectional", False)
    num_classes = config["model"]["num_classes"]
    dropout = config["model"]["dropout"]

    if model_type == "LSTM":
        return LSTMClassifier(
            vocab_size=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            pretrained_embeddings=torch.tensor(embedding_matrix, dtype=torch.float)
        )
    elif model_type == "GRU":
        return GRUClassifier(
            vocab_size=input_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            pretrained_embeddings=torch.tensor(embedding_matrix, dtype=torch.float)
        )
    elif model_type == "CNN":
        # Example for CNN, adjust hyperparameters if needed
        from src.models.neural_networks import CNNClassifier
        return CNNClassifier(
            vocab_size=input_dim,
            embedding_dim=embedding_dim,
            num_filters=128,
            filter_sizes=(3, 4, 5),
            num_classes=num_classes,
            dropout=dropout,
            pretrained_embeddings=torch.tensor(embedding_matrix, dtype=torch.float)
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# ----------------------------
# Main training pipeline
# ----------------------------
def main():
    # Load configs
    config, project_root = load_configs()
    print("✅ Loaded configs")

    # Load embeddings and preprocessed data
    embedding_matrix, X_train, y_train, X_test, y_test = load_embeddings_and_data(config)
    print(f"✅ Data loaded: X_train={X_train.shape}, X_test={X_test.shape}")

    # Build DataLoaders
    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test, batch_size=config["model"]["batch_size"]
    )

    # Build model
    model = build_model(config, embedding_matrix)
    print(f"✅ Model instantiated: {config['model_type']}")

    # Trainer
    trainer = NeuralNetworkTrainer(
        model=model,
        device="cuda",
        learning_rate=config["model"]["lr"]
    )

    # Train
    history = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=config["model"]["num_epochs"],
        save_path=config["model_save_path"]
    )

    print("✅ Training complete!")
    print(f"Best model saved to {config['model_save_path']}")

    # Save evaluation report
    report_path = Path("results/models/training_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    print(f"✅ Training report saved to {report_path}")


if __name__ == "__main__":
    main()
