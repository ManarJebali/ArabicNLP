"""
Complete feature extraction pipeline
"""

import numpy as np
from typing import List, Dict, Optional


class ArabicFeaturePipeline:
    """Unified feature extraction pipeline"""

    def __init__(self, method='embedding', **kwargs):
        """
        Args:
            method: 'statistical', 'embedding', 'tfidf_weighted', 'combined'
            **kwargs: Method-specific parameters
        """
        self.method = method
        self.kwargs = kwargs
        self.extractor = None

    def fit_transform(self, texts: List[str], token_sequences: List[List[int]],
                      labels: np.ndarray, embedding_matrix: Optional[np.ndarray] = None,
                      vocab: Optional[Dict[str, int]] = None) -> np.ndarray:
        """Extract features using chosen method"""

        if self.method == 'statistical':
            from src.features.statistical_selector import ArabicFeatureSelector
            from src.features.dimensionality_reducer import DimensionalityReducer

            selector_method = self.kwargs.get('selector', 'chi2')
            k = self.kwargs.get('k', 500)
            self.extractor = ArabicFeatureSelector(method=selector_method, k=k)
            X = self.extractor.fit_transform(texts, labels)

            if self.kwargs.get('reduce_dim', False):
                n_components = self.kwargs.get('n_components', 50)
                reducer = DimensionalityReducer(method='svd', n_components=n_components)
                X = reducer.fit_transform(X)

            return X

        elif self.method == 'embedding':
            from src.features.embedding_extractor import EmbeddingFeatureExtractor

            if embedding_matrix is None or vocab is None:
                raise ValueError("embedding_matrix and vocab required")

            pooling = self.kwargs.get('pooling', 'mean')
            use_pca = self.kwargs.get('use_pca', False)
            pca_dims = self.kwargs.get('pca_dims', 30)

            self.extractor = EmbeddingFeatureExtractor(
                embedding_matrix, vocab, pooling=pooling,
                use_pca=use_pca, pca_dims=pca_dims
            )
            return self.extractor.transform(token_sequences)

        elif self.method == 'tfidf_weighted':
            from src.features.embedding_extractor import TfidfWeightedEmbeddings

            if embedding_matrix is None or vocab is None:
                raise ValueError("embedding_matrix and vocab required")

            self.extractor = TfidfWeightedEmbeddings(embedding_matrix, vocab)
            return self.extractor.fit_transform(texts, token_sequences)

        elif self.method == 'combined':
            from src.features.statistical_selector import ArabicFeatureSelector
            from src.features.embedding_extractor import EmbeddingFeatureExtractor

            stat_selector = ArabicFeatureSelector(method='chi2', k=200)
            X_stat = stat_selector.fit_transform(texts, labels)

            emb_extractor = EmbeddingFeatureExtractor(
                embedding_matrix, vocab, pooling='mean'
            )
            X_emb = emb_extractor.transform(token_sequences)

            if hasattr(X_stat, 'toarray'):
                X_stat = X_stat.toarray()
            X = np.hstack([X_stat, X_emb])

            print(f"✓ Combined: {X_stat.shape[1]} stat + {X_emb.shape[1]} emb = {X.shape[1]} total")
            return X

        else:
            raise ValueError(f"Unknown method: {self.method}")
