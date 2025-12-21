"""
Train Word2Vec embeddings from processed data
Usage: python scripts/train_embeddings.py --data data/processed/final_list_train.pkl --vocab data/processed/vocab.pkl --output data/embeddings/
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embeddings.sgns_trainer import train_sgns_for_tokenize_outputs


def main():
    parser = argparse.ArgumentParser(description='Train Word2Vec Embeddings')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to tokenized sequences (pickle file)')
    parser.add_argument('--vocab', type=str, required=True,
                        help='Path to vocabulary (pickle file)')
    parser.add_argument('--output', type=str, default='data/embeddings/',
                        help='Output directory for embeddings')
    parser.add_argument('--embed-dim', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--window-size', type=int, default=2)
    parser.add_argument('--num-negatives', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING WORD2VEC EMBEDDINGS")
    print("=" * 70)

    # Load data
    print("\n1️⃣ Loading data...")
    with open(args.data, 'rb') as f:
        final_list_train = pickle.load(f)
    print(f"   ✓ Loaded {len(final_list_train)} sequences")

    with open(args.vocab, 'rb') as f:
        vocab = pickle.load(f)
    print(f"   ✓ Loaded vocabulary of size {len(vocab)}")

    # Train embeddings
    print("\n2️⃣ Training embeddings...")
    embedding_matrix, V = train_sgns_for_tokenize_outputs(
        final_list_train=final_list_train,
        onehot_dict=vocab,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr
    )

    print(f"\n   ✓ Embeddings shape: {embedding_matrix.shape}")

    # Save
    print("\n3️⃣ Saving embeddings...")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f'embeddings_{args.embed_dim}d.npy'
    np.save(output_path, embedding_matrix)
    print(f"   ✓ Saved to {output_path}")

    print("\n" + "=" * 70)
    print("✅ EMBEDDING TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
