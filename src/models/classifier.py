"""
Classification models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


class ArabicTextClassifier:
    """Simple text classifier wrapper"""

    def __init__(self, model_type='logistic', **kwargs):
        """
        Args:
            model_type: Type of classifier ('logistic', 'svm', 'naive_bayes')
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type

        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'svm':
            self.model = SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'linear'),
                probability=kwargs.get('probability', True),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB(
                alpha=kwargs.get('alpha', 1.0)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train classifier"""
        if hasattr(X_train, 'toarray'):
            X_train = X_train.toarray()
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if hasattr(X, 'toarray'):
            X = X.toarray()
        return self.model.predict(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> dict:
        """Evaluate model"""
        y_pred = self.predict(X)
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        return {
            'accuracy': acc,
            'predictions': y_pred,
            'report': report
        }
