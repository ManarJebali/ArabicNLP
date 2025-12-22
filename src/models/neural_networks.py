# ============================================================================
# FILE: src/models/neural_networks.py
# ============================================================================

"""
Neural Network models for Arabic text classification
Includes: RNN, LSTM, GRU, and CNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ============================================================================
# 1. LSTM Model
# ============================================================================

class LSTMClassifier(nn.Module):
    """
    LSTM-based text classifier
    Good for capturing long-term dependencies in sequences
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 bidirectional: bool = True,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False):
        """
        Args:
            vocab_size: Size of vocabulary (+ 1 for padding)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            pretrained_embeddings: Pre-trained embedding matrix
            freeze_embeddings: Whether to freeze embeddings during training
        """
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0
            )

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_len]

        Returns:
            Output logits [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # LSTM: output [batch_size, seq_len, hidden_dim * num_directions]
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Dropout and FC
        out = self.dropout(hidden)
        out = self.fc(out)

        return out


# ============================================================================
# 2. GRU Model
# ============================================================================

class GRUClassifier(nn.Module):
    """
    GRU-based text classifier
    Similar to LSTM but simpler and often faster
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 bidirectional: bool = True,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False):
        super(GRUClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0
            )

        # GRU layer
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)

        # GRU
        gru_out, hidden = self.gru(embedded)

        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]

        # Dropout and FC
        out = self.dropout(hidden)
        out = self.fc(out)

        return out


# ============================================================================
# 3. Simple RNN Model
# ============================================================================

class RNNClassifier(nn.Module):
    """
    Simple RNN-based text classifier
    Baseline model, less powerful than LSTM/GRU
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(RNNClassifier, self).__init__()

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0
            )

        # RNN layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)

        # Use last hidden state
        hidden = hidden[-1, :, :]

        out = self.dropout(hidden)
        out = self.fc(out)

        return out


# ============================================================================
# 4. CNN Model
# ============================================================================

class CNNClassifier(nn.Module):
    """
    CNN-based text classifier
    Good for capturing local n-gram patterns
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_filters: int,
                 filter_sizes: Tuple[int, ...],
                 num_classes: int,
                 dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None,
                 freeze_embeddings: bool = False):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            num_filters: Number of filters per filter size
            filter_sizes: Tuple of filter sizes (e.g., (3, 4, 5))
            num_classes: Number of output classes
            dropout: Dropout probability
            pretrained_embeddings: Pre-trained embeddings
            freeze_embeddings: Freeze embeddings during training
        """
        super(CNNClassifier, self).__init__()

        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0
            )

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=fs
            )
            for fs in filter_sizes
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len]
        Returns:
            [batch_size, num_classes]
        """
        # Embedding: [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # Transpose for conv1d: [batch_size, embedding_dim, seq_len]
        embedded = embedded.permute(0, 2, 1)

        # Apply convolution and max pooling for each filter size
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        # Concatenate pooled features
        cat = torch.cat(pooled, dim=1)

        # Dropout and FC
        out = self.dropout(cat)
        out = self.fc(out)

        return out


# ============================================================================
# 5. Hybrid CNN-LSTM Model
# ============================================================================

class CNN_LSTM_Classifier(nn.Module):
    """
    Hybrid CNN-LSTM model
    CNN extracts local features, LSTM captures sequence information
    """

    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 num_filters: int,
                 filter_sizes: Tuple[int, ...],
                 lstm_hidden_dim: int,
                 num_classes: int,
                 dropout: float = 0.5,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(CNN_LSTM_Classifier, self).__init__()

        # Embedding
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size + 1,
                embedding_dim,
                padding_idx=0
            )

        # CNN layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])

        # LSTM layer
        cnn_output_dim = len(filter_sizes) * num_filters
        self.lstm = nn.LSTM(
            cnn_output_dim,
            lstm_hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # FC layer
        self.fc = nn.Linear(lstm_hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)

        # CNN
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)

        # Reshape for LSTM: [batch_size, 1, features]
        cat = cat.unsqueeze(1)

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(cat)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # FC
        out = self.dropout(hidden)
        out = self.fc(out)

        return out
