"""
Embedding-based feature extraction with pooling
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List, Dict, Optional


class EmbeddingFeatureExtractor:
    """Extract features from word embeddings"""

    def __init__(self, embedding_matrix: np.ndarray, vocab: Dict[str, int],
                 pooling='mean', use_pca=False, pca_dims=None):
        """
        Args:
            embedding_matrix: Shape [vocab_size+1, embed_dim]
            vocab: Word to ID mapping (1-indexed)
            pooling: 'mean', 'max', 'sum', or 'weighted'
            use_pca: Whether to apply PCA
            pca_dims: Number of PCA dimensions
        """
        self.embedding_matrix = embedding_matrix
        self.vocab = vocab
        self.pooling = pooling
        self.use_pca = use_pca
        self.pca_dims = pca_dims
        self.pca = None
        self.embed_dim = embedding_matrix.shape[1]

    def _pool_embeddings(self, token_ids: List[int],
                         weights: Optional[List[float]] = None) -> np.ndarray:
        """Pool embeddings for a sequence"""
        if len(token_ids) == 0:
            return np.zeros(self.embed_dim)

        embeddings = self.embedding_matrix[token_ids]

        if self.pooling == 'mean':
            return embeddings.mean(axis=0)
        elif self.pooling == 'max':
            return embeddings.max(axis=0)
        elif self.pooling == 'sum':
            return embeddings.sum(axis=0)
        elif self.pooling == 'weighted':
            if weights is None:
                weights = np.ones(len(token_ids))
            weights = np.array(weights).reshape(-1, 1)
            return (embeddings * weights).sum(axis=0) / weights.sum()
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

    def transform(self, token_sequences: List[List[int]],
                  tfidf_weights: Optional[List[List[float]]] = None) -> np.ndarray:
        """Transform token sequences to features"""
        features = []
        for i, seq in enumerate(token_sequences):
            weights = tfidf_weights[i] if tfidf_weights else None
            features.append(self._pool_embeddings(seq, weights))

        X = np.array(features)

        if self.use_pca and self.pca_dims:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_dims, random_state=42)
                X = self.pca.fit_transform(X)
                var_explained = self.pca.explained_variance_ratio_.sum()
                print(f"✓ Applied PCA: {self.embed_dim}D → {self.pca_dims}D")
                print(f"  Variance explained: {var_explained:.2%}")
            else:
                X = self.pca.transform(X)

        return X


class TfidfWeightedEmbeddings:
    """TF-IDF weighted embeddings"""

    def __init__(self, embedding_matrix: np.ndarray, vocab: Dict[str, int]):
        self.embedding_matrix = embedding_matrix
        self.vocab = vocab
        self.tfidf_vectorizer = None
        self.id_to_word = {v: k for k, v in vocab.items()}

    def fit_transform(self, texts: List[str],
                      token_sequences: List[List[int]]) -> np.ndarray:
        """Compute TF-IDF weighted embeddings"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.tfidf_vectorizer = TfidfVectorizer(vocabulary=self.vocab)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)

        features = []
        for i, seq in enumerate(token_sequences):
            if len(seq) == 0:
                features.append(np.zeros(self.embedding_matrix.shape[1]))
                continue

            doc_tfidf = tfidf_matrix[i].toarray().flatten()
            weighted_sum = np.zeros(self.embedding_matrix.shape[1])
            total_weight = 0

            for token_id in seq:
                word = self.id_to_word.get(token_id)
                if word and word in self.vocab:
                    tfidf_weight = doc_tfidf[self.vocab[word] - 1]
                    weighted_sum += self.embedding_matrix[token_id] * tfidf_weight
                    total_weight += tfidf_weight

            if total_weight > 0:
                features.append(weighted_sum / total_weight)
            else:
                features.append(weighted_sum)

        return np.array(features)
