import re
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download Arabic stopwords
nltk.download('stopwords')

# ==========================================
# STEP 1: Minimal Cleaning for Arabic Text
# ==========================================

# Precompiled patterns
URL_PATTERN = re.compile(r'\b(?:https?://|http://|www\.)\S+\b', re.IGNORECASE)
EMAIL_PATTERN = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
MENTION_PATTERN = re.compile(r'@\w+')
NUMBER_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')

# Arabic-specific patterns
ARABIC_DIACRITICS = re.compile(r'[\u064B-\u065F\u0670]')  # Tashkeel/diacritics
TATWEEL = re.compile(r'\u0640')  # Tatweel (ـ)

# Normalize Arabic characters
ARABIC_NORMALIZE_MAP = {
    'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا',  # Alef variations
    'ة': 'ه',  # Taa Marbuta to Haa
    'ى': 'ي',  # Alef Maksura to Yaa
}

TRANSLATE_MAP = {
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...",
    "\u00A0": " ",
}

def _normalize_unicode(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    return s.translate(str.maketrans(TRANSLATE_MAP))

def _remove_control_chars(s: str) -> str:
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("C"))

def _normalize_arabic(s: str) -> str:
    """Normalize Arabic-specific characters"""
    # Remove diacritics (tashkeel)
    s = ARABIC_DIACRITICS.sub('', s)
    # Remove tatweel
    s = TATWEEL.sub('', s)
    # Normalize Alef and other variations
    for old, new in ARABIC_NORMALIZE_MAP.items():
        s = s.replace(old, new)
    return s

def _replace_entities(s: str, replace_numbers: bool = True) -> str:
    s = EMAIL_PATTERN.sub("<EMAIL>", s)
    s = URL_PATTERN.sub("<URL>", s)
    s = MENTION_PATTERN.sub("<USER>", s)
    if replace_numbers:
        # Also replace Arabic-Indic digits (٠-٩)
        s = re.sub(r'[\u0660-\u0669]', '<NUM>', s)
        s = NUMBER_PATTERN.sub("<NUM>", s)
    return s

def _normalize_whitespace(s: str) -> str:
    s = WHITESPACE_PATTERN.sub(" ", s)
    return s.strip()

def clean_arabic_text(text: str, replace_numbers: bool = True) -> str:
    """
    Minimal cleaning for Arabic text:
    - Keep punctuation and emojis
    - Replace entities: URLs, emails, mentions, numbers
    - Normalize Arabic characters (remove diacritics, normalize Alef variants)
    - Remove control characters
    - Normalize whitespace
    """
    if not isinstance(text, str):
        return ""
    s = _normalize_unicode(text)
    s = _remove_control_chars(s)
    s = _normalize_arabic(s)
    s = _replace_entities(s, replace_numbers=replace_numbers)
    s = _normalize_whitespace(s)
    return s

def apply_arabic_cleaning(df: pd.DataFrame, text_col: str = "tweet") -> pd.DataFrame:
    """Apply Arabic cleaning to DataFrame"""
    df["text"] = df[text_col].apply(clean_arabic_text)
    return df
