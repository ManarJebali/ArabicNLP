"""
Dimensionality reduction using PCA and TruncatedSVD
"""

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD


class DimensionalityReducer:
    """Dimensionality reduction"""

    def __init__(self, method='pca', n_components=50):
        """
        Args:
            method: 'pca' or 'svd'
            n_components: number of components
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform data"""
        if self.method == 'pca':
            if hasattr(X, 'toarray'):
                X = X.toarray()
            self.reducer = PCA(n_components=self.n_components, random_state=42)
        elif self.method == 'svd':
            self.reducer = TruncatedSVD(n_components=self.n_components, random_state=42)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        X_reduced = self.reducer.fit_transform(X)

        if hasattr(self.reducer, 'explained_variance_ratio_'):
            var_explained = self.reducer.explained_variance_ratio_.sum()
            print(f"✓ Reduced to {self.n_components}D using {self.method.upper()}")
            print(f"  Variance explained: {var_explained:.2%}")

        return X_reduced

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data"""
        if self.reducer is None:
            raise ValueError("Must call fit_transform first")
        if self.method == 'pca' and hasattr(X, 'toarray'):
            X = X.toarray()
        return self.reducer.transform(X)