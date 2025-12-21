"""
Statistical feature selection (Chi-square, Mutual Information)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from typing import List
import warnings

warnings.filterwarnings('ignore')


class ArabicFeatureSelector:
    """Statistical feature selection for Arabic text"""

    def __init__(self, method='chi2', k=500):
        """
        Args:
            method: 'chi2' or 'mutual_info'
            k: number of top features to select
        """
        self.method = method
        self.k = k
        self.selector = None
        self.vectorizer = None
        self.selected_features = None

    def fit_transform(self, texts: List[str], labels: np.ndarray) -> np.ndarray:
        """Fit selector and transform data"""
        # Create TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2
        )
        X = self.vectorizer.fit_transform(texts)

        # Select scoring function
        if self.method == 'chi2':
            score_func = chi2
        elif self.method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Fit selector
        self.selector = SelectKBest(score_func=score_func, k=min(self.k, X.shape[1]))
        X_selected = self.selector.fit_transform(X, labels)

        # Get selected feature names
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        selected_mask = self.selector.get_support()
        self.selected_features = feature_names[selected_mask]

        print(f"✓ Selected {X_selected.shape[1]} features using {self.method}")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Top 10 features: {list(self.selected_features[:10])}")

        return X_selected

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts"""
        if self.vectorizer is None or self.selector is None:
            raise ValueError("Must call fit_transform first")
        X = self.vectorizer.transform(texts)
        return self.selector.transform(X)

    def get_feature_scores(self, top_n=20) -> pd.DataFrame:
        """Get scores for top N features"""
        if self.selector is None:
            raise ValueError("Must call fit_transform first")

        feature_names = np.array(self.vectorizer.get_feature_names_out())
        scores = self.selector.scores_
        selected_mask = self.selector.get_support()

        df = pd.DataFrame({
            'feature': feature_names[selected_mask],
            'score': scores[selected_mask]
        })
        return df.sort_values('score', ascending=False).head(top_n)

