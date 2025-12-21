"""
End-to-end pipeline script for Arabic NLP
Usage: python scripts/run_pipeline.py --config configs/default_config.yaml
"""

import argparse
import yaml
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

from src.utils.data_loader import load_arabic_dataset
from src.preprocessing.arabic_cleaner import apply_arabic_cleaning
from src.preprocessing.tokenizer import tokenize_arabic, padding_
from src.embeddings.sgns_trainer import train_sgns_for_tokenize_outputs
from src.features.pipeline import ArabicFeaturePipeline


def setup_logger(results_dir):
    """Setup logger for pipeline"""
    logger = logging.getLogger("arabic_nlp_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # File handler
        log_path = os.path.join(results_dir, "logs", "pipeline.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


def main(config_path):
    """Main pipeline function"""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup directories
    results_dir = config.get('results_dir', 'results')
    processed_dir = 'data/processed'
    embeddings_dir = 'data/embeddings'

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'reports'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'logs'), exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    logger = setup_logger(results_dir)

    logger.info("=" * 70)
    logger.info("ARABIC NLP PIPELINE - STARTING")
    logger.info("=" * 70)

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    logger.info("\n1️⃣ Loading data...")
    train_path = config['data']['train_path']
    logger.info(f"   Data path: {train_path}")

    df = load_arabic_dataset(train_path, text_col='tweet', label_col='label')
    logger.info(f"   ✓ Loaded {len(df)} samples")

    # ========================================================================
    # STEP 2: Clean Text
    # ========================================================================
    logger.info("\n2️⃣ Cleaning text...")
    df = apply_arabic_cleaning(df, text_col='tweet')
    logger.info(f"   ✓ Cleaned {len(df)} texts")

    # ========================================================================
    # STEP 3: Train-Test Split
    # ========================================================================
    logger.info("\n3️⃣ Splitting data...")
    test_size = config['data'].get('test_size', 0.2)
    random_state = config['data'].get('random_state', 42)

    x_train, x_test, y_train, y_test = train_test_split(
        df['text'].values,
        df['label'].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label'].values
    )
    logger.info(f"   ✓ Train: {len(x_train)} samples")
    logger.info(f"   ✓ Test: {len(x_test)} samples")

    # ========================================================================
    # STEP 4: Tokenization
    # ========================================================================
    logger.info("\n4️⃣ Tokenizing...")
    vocab_size = config['preprocessing'].get('vocab_size', 1000)

    final_list_train, encoded_train, final_list_test, encoded_test, vocab = tokenize_arabic(
        x_train, y_train, x_test, y_test, vocab_size=vocab_size
    )
    logger.info(f"   ✓ Vocabulary size: {len(vocab)}")

    # ========================================================================
    # STEP 5: Padding (if needed for LSTM/RNN)
    # ========================================================================
    logger.info("\n5️⃣ Padding sequences...")
    max_seq_len = config['preprocessing'].get('max_seq_len', 50)

    x_train_pad = padding_(final_list_train, seq_len=max_seq_len)
    x_test_pad = padding_(final_list_test, seq_len=max_seq_len)
    logger.info(f"   ✓ Padded to length: {max_seq_len}")
    logger.info(f"   ✓ Train shape: {x_train_pad.shape}")
    logger.info(f"   ✓ Test shape: {x_test_pad.shape}")

    # ========================================================================
    # STEP 6: Save Processed Data
    # ========================================================================
    logger.info("\n6️⃣ Saving processed data...")

    # Save padded sequences
    np.save(os.path.join(processed_dir, 'x_train_pad.npy'), x_train_pad)
    np.save(os.path.join(processed_dir, 'y_train.npy'), encoded_train)
    np.save(os.path.join(processed_dir, 'x_test_pad.npy'), x_test_pad)
    np.save(os.path.join(processed_dir, 'y_test.npy'), encoded_test)

    # Save token sequences (un-padded)
    with open(os.path.join(processed_dir, 'final_list_train.pkl'), 'wb') as f:
        pickle.dump(final_list_train, f)
    with open(os.path.join(processed_dir, 'final_list_test.pkl'), 'wb') as f:
        pickle.dump(final_list_test, f)

    # Save text data
    pd.DataFrame({'text': x_train, 'label': y_train}).to_csv(
        os.path.join(processed_dir, 'train.csv'), index=False
    )
    pd.DataFrame({'text': x_test, 'label': y_test}).to_csv(
        os.path.join(processed_dir, 'test.csv'), index=False
    )

    # Save vocabulary
    with open(os.path.join(processed_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump(vocab, f)

    logger.info(f"   ✓ Saved to {processed_dir}/")

    # ========================================================================
    # STEP 7: Train Word2Vec Embeddings
    # ========================================================================
    logger.info("\n7️⃣ Training Word2Vec embeddings...")

    embedding_matrix, V = train_sgns_for_tokenize_outputs(
        final_list_train=final_list_train,
        onehot_dict=vocab,
        embed_dim=config['embeddings']['embed_dim'],
        window_size=config['embeddings']['window_size'],
        num_negatives=config['embeddings']['num_negatives'],
        batch_size=config['embeddings']['batch_size'],
        epochs=config['embeddings']['epochs'],
        lr=config['embeddings']['learning_rate']
    )

    logger.info(f"   ✓ Embeddings shape: {embedding_matrix.shape}")

    # Save embeddings
    embed_dim = config['embeddings']['embed_dim']
    np.save(os.path.join(embeddings_dir, f'embeddings_{embed_dim}d.npy'), embedding_matrix)
    logger.info(f"   ✓ Saved to {embeddings_dir}/embeddings_{embed_dim}d.npy")

    # ========================================================================
    # STEP 8: Feature Extraction
    # ========================================================================
    logger.info("\n8️⃣ Extracting features...")

    pipeline = ArabicFeaturePipeline(
        method=config['features']['method'],
        pooling=config['features'].get('pooling', 'mean'),
        use_pca=config['features'].get('use_pca', False),
        pca_dims=config['features'].get('pca_dims', 30)
    )

    X_train_feat = pipeline.fit_transform(
        x_train, final_list_train, y_train,
        embedding_matrix=embedding_matrix,
        vocab=vocab
    )
    X_test_feat = pipeline.extractor.transform(final_list_test)

    logger.info(f"   ✓ Train features shape: {X_train_feat.shape}")
    logger.info(f"   ✓ Test features shape: {X_test_feat.shape}")

    # Save features
    if hasattr(X_train_feat, 'toarray'):
        X_train_feat = X_train_feat.toarray()
    if hasattr(X_test_feat, 'toarray'):
        X_test_feat = X_test_feat.toarray()

    np.save(os.path.join(processed_dir, 'X_train_features.npy'), X_train_feat)
    np.save(os.path.join(processed_dir, 'X_test_features.npy'), X_test_feat)
    logger.info(f"   ✓ Saved features to {processed_dir}/")

    # ========================================================================
    # STEP 9: Train Classifier
    # ========================================================================
    logger.info("\n9️⃣ Training classifier...")

    model_config = config.get('model', {})
    model = LogisticRegression(
        max_iter=model_config.get('max_iter', 1000),
        random_state=random_state
    )
    model.fit(X_train_feat, y_train)
    logger.info("   ✓ Classifier trained")

    # ========================================================================
    # STEP 10: Evaluation
    # ========================================================================
    logger.info("\n🔟 Evaluating model...")

    y_pred_train = model.predict(X_train_feat)
    y_pred_test = model.predict(X_test_feat)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    train_report = classification_report(y_train, y_pred_train, digits=4)
    test_report = classification_report(y_test, y_pred_test, digits=4)

    logger.info(f"\n   Train Accuracy: {train_acc:.4f}")
    logger.info(f"   Test Accuracy:  {test_acc:.4f}")

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT - TRAIN SET")
    print("=" * 70)
    print(train_report)

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT - TEST SET")
    print("=" * 70)
    print(test_report)

    # ========================================================================
    # STEP 11: Save Results
    # ========================================================================
    logger.info("\n💾 Saving results...")

    # Save model
    model_path = os.path.join(results_dir, 'models', 'best_model.pkl')
    joblib.dump(model, model_path)
    logger.info(f"   ✓ Model saved to {model_path}")

    # Save reports
    with open(os.path.join(results_dir, 'reports', 'train_report.txt'), 'w') as f:
        f.write(f"Train Accuracy: {train_acc:.4f}\n\n")
        f.write(train_report)

    with open(os.path.join(results_dir, 'reports', 'test_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write(test_report)

    logger.info(f"   ✓ Reports saved to {results_dir}/reports/")

    # Save metadata
    metadata = {
        'vocab_size': len(vocab),
        'embed_dim': embed_dim,
        'max_seq_len': max_seq_len,
        'train_samples': len(x_train),
        'test_samples': len(x_test),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'config': config
    }

    with open(os.path.join(results_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump(metadata, f)

    logger.info("=" * 70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"\nResults saved to: {results_dir}/")
    logger.info(f"Processed data saved to: {processed_dir}/")
    logger.info(f"Embeddings saved to: {embeddings_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run complete Arabic NLP pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    args = parser.parse_args()
    main(args.config)
