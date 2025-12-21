"""
End-to-end pipeline script for Arabic NLP
Run as: python scripts/run_pipeline.py --config configs/default_config.yaml
"""

import argparse
import yaml

from src.utils.data_loader import load_arabic_dataset
from src.preprocessing.padding import padding_
from src.preprocessing.arabic_cleaner import clean_arabic_text
# Clean imports from anywhere in the project
from src.preprocessing.arabic_cleaner import clean_arabic_text
from src.preprocessing.tokenizer import tokenize_arabic
from src.embeddings.sgns_trainer import train_sgns_for_tokenize_outputs
from src.features.embedding_extractor import EmbeddingFeatureExtractor
from src.features.pipeline import ArabicFeaturePipeline

def main(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger = setup_logger()

    # 1. Load Data
    logger.info("Loading data...")
    df = load_data(config['data']['train_path'])

    # 2. Clean Text
    logger.info("Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_arabic_text)

    # 3. Tokenize
    logger.info("Tokenizing text...")
    df['tokens'] = df['cleaned_text'].apply(tokenize_arabic)

    # 4. Pad Sequences
    logger.info("Padding sequences...")
    maxlen = config['preprocessing'].get('max_seq_len', 50)
    df['padded_tokens'] = pad_sequences(
        df['tokens'], maxlen=maxlen, padding='post', truncating='post', value=0
    )

    # 5. Train embeddings
    logger.info("Training embeddings...")
    embedding_matrix, vocab = train_sgns_for_tokenize_outputs(
        df['tokens'],
        embed_dim=config['embeddings']['embed_dim'],
        window=config['embeddings']['window_size'],
        epochs=config['embeddings']['epochs']
    )

    # 6. Feature Extraction
    logger.info("Extracting features...")
    pipeline = ArabicFeaturePipeline(
        config=config['features'],
        embedding_matrix=embedding_matrix,
        vocab=vocab
    )
    X = pipeline.transform(df['padded_tokens'])
    y = df['label']

    # 7. Train model
    logger.info("Training model...")
    model = train_classifier(X, y, config=config['model'])

    # 8. Evaluate
    logger.info("Evaluating...")
    metrics = evaluate_model(model, X, y)
    print(metrics)
    logger.info("Pipeline completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML"
    )
    args = parser.parse_args()
    main(args.config)