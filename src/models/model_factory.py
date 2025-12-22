import torch
from src.models.Simple_forwardNet import FeedForwardNN
from src.models.neural_networks import (
    LSTMClassifier,
    GRUClassifier,
    CNNClassifier
)


def build_model(config, embeddings=None):
    model_type = config["model_type"]
    mcfg = config["model"]

    if model_type == "Simple_FeedForwardNN":
        return FeedForwardNN(
            input_dim=mcfg["input_dim"],
            hidden_dim=mcfg["hidden_dim"],
            num_classes=mcfg["num_classes"]
        )

    elif model_type == "LSTM":
        return LSTMClassifier(
            vocab_size=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_classes=mcfg["num_classes"],
            dropout=mcfg["dropout"],
            bidirectional=mcfg["bidirectional"],
            pretrained_embeddings=torch.tensor(embeddings, dtype=torch.float32)
        )

    elif model_type == "GRU":
        return GRUClassifier(
            vocab_size=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            num_classes=mcfg["num_classes"],
            dropout=mcfg["dropout"],
            bidirectional=mcfg["bidirectional"],
            pretrained_embeddings=torch.tensor(embeddings, dtype=torch.float32)
        )

    elif model_type == "CNN":
        return CNNClassifier(
            vocab_size=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            num_filters=100,
            filter_sizes=(3, 4, 5),
            num_classes=mcfg["num_classes"],
            dropout=mcfg["dropout"],
            pretrained_embeddings=torch.tensor(embeddings, dtype=torch.float32)
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
