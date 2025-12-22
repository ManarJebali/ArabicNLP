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
def load_config():
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "model_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config, project_root


# ----------------------------
# Load TF-IDF features
# ----------------------------
def load_tfidf_features(processed_dir: Path):
    X_train = np.load(processed_dir / "X_train_tfidf.npy", allow_pickle=True)
    y_train = np.load(processed_dir / "y_train.npy", allow_pickle=True)
    X_test  = np.load(processed_dir / "X_test_tfidf.npy", allow_pickle=True)
    y_test  = np.load(processed_dir / "y_test.npy", allow_pickle=True)

    # ✅ FIX: force numeric labels
    y_train = np.asarray(y_train, dtype=np.int64)
    y_test  = np.asarray(y_test, dtype=np.int64)

    return X_train, y_train, X_test, y_test


# ----------------------------
# Build DataLoaders
# ----------------------------
def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ----------------------------
# Training loop
# ----------------------------
def train_model(model, train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["lr"])

    best_acc = 0.0
    save_path = Path(config["model_save_path"])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["model"]["num_epochs"]):
        model.train()
        correct, total, loss_sum = 0, 0, 0.0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                correct += (outputs.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ Best validation accuracy: {best_acc:.4f}")
    print(f"💾 Model saved to: {save_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    config, project_root = load_config()

    processed_dir = project_root / config["data"]["processed_dir"]

    X_train, y_train, X_test, y_test = load_tfidf_features(processed_dir)

    config["model"]["input_dim"] = X_train.shape[1]

    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test,
        batch_size=config["model"]["batch_size"]
    )

    model = FeedForwardNN(
        input_dim=config["model"]["input_dim"],
        hidden_dim=config["model"]["hidden_dim"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"]
    )

    train_model(model, train_loader, test_loader, config)


if __name__ == "__main__":
    main()
