"""
Arabic NLP Package
"""

__version__ = '0.1.0'

from src.preprocessing.arabic_cleaner import clean_arabic_text, apply_arabic_cleaning
from src.preprocessing.tokenizer import tokenize_arabic
from src.preprocessing.padding import padding_

from src.embeddings.sgns_trainer import train_sgns_for_tokenize_outputs
from src.features.pipeline import ArabicFeaturePipeline
from src.models.classifier import ArabicTextClassifier

__all__ = [
    'clean_arabic_text',
    'apply_arabic_cleaning',
    'tokenize_arabic',
    'train_sgns_for_tokenize_outputs',
    'ArabicFeaturePipeline',
    'ArabicTextClassifier',
]