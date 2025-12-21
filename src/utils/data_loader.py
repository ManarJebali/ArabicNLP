"""
Data loading utilities
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


def create_dataloaders(x_train_pad: np.ndarray,
                       y_train: np.ndarray,
                       x_test_pad: np.ndarray,
                       y_test: np.ndarray,
                       batch_size: int = 50,
                       shuffle: bool = True) -> tuple:
    """
    Create PyTorch DataLoaders from padded sequences

    Args:
        x_train_pad: Padded training sequences
        y_train: Training labels
        x_test_pad: Padded test sequences
        y_test: Test labels
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle training data

    Returns:
        train_loader, valid_loader
    """
    # Create Tensor datasets
    train_data = TensorDataset(
        torch.from_numpy(x_train_pad),
        torch.from_numpy(y_train)
    )
    valid_data = TensorDataset(
        torch.from_numpy(x_test_pad),
        torch.from_numpy(y_test)
    )

    # Create dataloaders
    train_loader = DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=shuffle, batch_size=batch_size)

    return train_loader, valid_loader


def load_arabic_dataset(file_path: str, text_col: str = 'tweet',
                        label_col: str = 'label') -> pd.DataFrame:
    """
    Load Arabic dataset from file

    Args:
        file_path: Path to CSV or Excel file
        text_col: Name of text column
        label_col: Name of label column

    Returns:
        DataFrame with text and labels
    """
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("File must be .csv or .xlsx")

    print(f"✓ Loaded {len(df)} samples")
    print(f"  Columns: {df.columns.tolist()}")

    if label_col in df.columns:
        print(f"  Label distribution:")
        print(df[label_col].value_counts())

    return df