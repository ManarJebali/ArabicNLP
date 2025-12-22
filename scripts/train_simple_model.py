import os
import json
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

from src.models.Simple_forwardNet import FeedForwardNN


# ----------------------------
# Config loader
# ----------------------------
def load_config(config_path="configs/model_config.yaml"):
    """
    Load YAML configuration
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config, project_root


# ----------------------------
# Load TF-IDF features
# ----------------------------
def load_tfidf_features(processed_dir: Path):
    files = {
        "X_train": processed_dir / "X_train_tfidf.npy",
        "y_train": processed_dir / "y_train.npy",
        "X_test":  processed_dir / "X_test_tfidf.npy",
        "y_test":  processed_dir / "y_test.npy"
    }

    for name, path in files.items():
        if not path.exists():
            print(f"Available files in {processed_dir}:")
            print(list(processed_dir.iterdir()))
            raise FileNotFoundError(f"Missing file: {path}")

    X_train = np.load(files["X_train"], allow_pickle=True)
    y_train = np.load(files["y_train"], allow_pickle=True)
    X_test  = np.load(files["X_test"], allow_pickle=True)
    y_test  = np.load(files["y_test"], allow_pickle=True)

    return X_train, y_train, X_test, y_test


# ----------------------------
# Build DataLoaders
# ----------------------------
def make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


# ----------------------------
# Training function
# ----------------------------
def train_model(model, train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"])
    num_epochs = config["model"]["num_epochs"]

    best_acc = 0.0
    save_path = Path(config["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                outputs = model(X_val)
                _, predicted = torch.max(outputs, 1)
                total_val += y_val.size(0)
                correct_val += (predicted == y_val).sum().item()
        val_acc = correct_val / total_val

        print(f"Epoch [{epoch}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} "
              f"Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"✅ Training complete! Best validation accuracy: {best_acc:.4f}")
    print(f"Model saved to: {save_path}")


# ----------------------------
# Main pipeline
# ----------------------------
def main():
    # Load config and project root
    config, project_root = load_config()

    # Absolute processed folder path
    processed_dir = project_root / "data" / "processed"

    # Load TF-IDF features
    X_train, y_train, X_test, y_test = load_tfidf_features(processed_dir)

    # Update input_dim based on loaded TF-IDF
    config["model"]["input_dim"] = X_train.shape[1]

    # Build DataLoaders
    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test, batch_size=config["model"]["batch_size"]
    )

    # Instantiate model
    model = FeedForwardNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    )

    # Train the model
    train_model(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
