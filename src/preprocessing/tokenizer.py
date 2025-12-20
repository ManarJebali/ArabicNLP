from nltk.corpus import stopwords

import re
import unicodedata
import pandas as pd
import numpy as np
from collections import Counter
import nltk

from src.preprocessing.arabic_cleaner import apply_arabic_cleaning


def preprocess_arabic_string(s):
    """
    Preprocess Arabic string:
    - Remove punctuation
    - Remove extra whitespace
    - Remove digits
    - Keep only Arabic letters
    """
    # Keep only Arabic letters, spaces
    s = re.sub(r'[^\u0600-\u06FF\s]', '', s)
    # Replace all runs of whitespaces with single space
    s = re.sub(r"\s+", ' ', s)
    return s.strip()


def tokenize_arabic(x_train, y_train, x_val, y_val, vocab_size=1000):
    """
    Tokenize Arabic text for training and validation sets

    Args:
        x_train: Training texts (list of strings)
        y_train: Training labels
        x_val: Validation texts (list of strings)
        y_val: Validation labels
        vocab_size: Maximum vocabulary size (default: 1000)

    Returns:
        final_list_train: List of token ID sequences for training
        encoded_train: Encoded training labels
        final_list_test: List of token ID sequences for validation
        encoded_test: Encoded validation labels
        onehot_dict: Dictionary mapping words to IDs
    """
    word_list = []

    # Load Arabic stopwords
    try:
        stop_words = set(stopwords.words('arabic'))
    except:
        print("Arabic stopwords not found. Using empty set.")
        stop_words = set()

    # Build vocabulary from training data
    for sent in x_train:
        if not isinstance(sent, str):
            continue
        for word in sent.split():
            word = preprocess_arabic_string(word)
            if word and word not in stop_words and len(word) > 1:
                word_list.append(word)

    # Count word frequencies
    corpus = Counter(word_list)
    # Get top N most common words
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:vocab_size]
    # Create word-to-id mapping (1-indexed)
    onehot_dict = {w: i+1 for i, w in enumerate(corpus_)}

    print(f"Vocabulary size: {len(onehot_dict)}")
    print(f"Most common words: {corpus_[:10]}")

    # Tokenize training data
    final_list_train = []
    for sent in x_train:
        if not isinstance(sent, str):
            final_list_train.append([])
            continue
        tokens = [onehot_dict[preprocess_arabic_string(word)]
                 for word in sent.split()
                 if preprocess_arabic_string(word) in onehot_dict.keys()]
        final_list_train.append(tokens)

    # Tokenize validation data
    final_list_test = []
    for sent in x_val:
        if not isinstance(sent, str):
            final_list_test.append([])
            continue
        tokens = [onehot_dict[preprocess_arabic_string(word)]
                 for word in sent.split()
                 if preprocess_arabic_string(word) in onehot_dict.keys()]
        final_list_test.append(tokens)

    encoded_train = np.array(y_train)
    encoded_test = np.array(y_val)

    return final_list_train, encoded_train, final_list_test, encoded_test, onehot_dict


# ==========================================
# EXAMPLE USAGE
# ==========================================

if __name__ == "__main__":
    # Example with your data structure
    # Assuming you have a DataFrame 'df' with columns: tweet, label, etc.

    # Sample data based on your screenshot
    sample_data = {
        'tweet': [
            'والله شايفينك الناس كلها صابر يا عبد',
            'أكيد اكتتاب ي زينه شاء الله الحين',
            'والله انا ما انا عارف اكتتاب ولا انطلاء ولا',
        ],
        'label': [1, 1, 1]
    }

    df = pd.DataFrame(sample_data)

    # Step 1: Apply cleaning
    df_clean = apply_arabic_cleaning(df.copy(), text_col='tweet')
    print("Cleaned text:")
    print(df_clean['text'].head())
    print()

    # Step 2: Split data
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        df_clean['text'].values,
        df_clean['label'].values,
        test_size=0.2,
        random_state=42
    )

    # Step 3: Tokenize
    final_list_train, encoded_train, final_list_test, encoded_test, vocab = tokenize_arabic(
        x_train, y_train, x_val, y_val, vocab_size=1000
    )

    print(f"Training sequences: {len(final_list_train)}")
    print(f"Validation sequences: {len(final_list_test)}")
    print(f"Sample tokenized sequence: {final_list_train[0]}")