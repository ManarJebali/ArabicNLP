"""
Utility functions for working with embeddings.
"""

import numpy as np

def save_embedding_matrix(matrix: np.ndarray, path: str):
    """
    Save embedding matrix to a .npy file.
    """
    np.save(path, matrix)

def load_embedding_matrix(path: str) -> np.ndarray:
    """
    Load embedding matrix from a .npy file.
    """
    return np.load(path)

def save_embedding_metadata(metadata: dict, path: str):
    """
    Save embedding metadata (such as word-to-id dict) to JSON.
    """
    import json
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_embedding_metadata(path: str) -> dict:
    """
    Load embedding metadata from JSON.
    """
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)