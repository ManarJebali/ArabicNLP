"""
Train classifier on extracted features
Usage: python scripts/train_classifier.py --features data/processed/X_train_features.npy --labels data/processed/y_train.npy --output results/models/classifier.pkl
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def main():
    parser = argparse.ArgumentParser(description='Train Classifier')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to training features (npy)')
    parser.add_argument('--labels', type=str, required=True,
                        help='Path to training labels (npy)')
    parser.add_argument('--test-features', type=str, default=None,
                        help='Path to test features (npy) - optional')
    parser.add_argument('--test-labels', type=str, default=None,
                        help='Path to test labels (npy) - optional')
    parser.add_argument('--output', type=str, default='results/models/classifier.pkl',
                        help='Output path for trained model')
    parser.add_argument('--max-iter', type=int, default=1000)
    args = parser.parse_args()

    print("=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    # Load training data
    print("\n1️⃣ Loading training data...")
    X_train = np.load(args.features)
    y_train = np.load(args.labels)
    print(f"   ✓ Training features: {X_train.shape}")
    print(f"   ✓ Training labels: {y_train.shape}")

    # Load test data if provided
    if args.test_features and args.test_labels:
        print("\n2️⃣ Loading test data...")
        X_test = np.load(args.test_features)
        y_test = np.load(args.test_labels)
        print(f"   ✓ Test features: {X_test.shape}")
        print(f"   ✓ Test labels: {y_test.shape}")
    else:
        X_test, y_test = None, None

    # Train
    print("\n3️⃣ Training classifier...")
    clf = LogisticRegression(max_iter=args.max_iter, random_state=42)
    clf.fit(X_train, y_train)
    print("   ✓ Classifier trained")

    # Evaluate
    print("\n4️⃣ Evaluating...")
    y_pred_train = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    print(f"   Train Accuracy: {train_acc:.4f}")

    if X_test is not None and y_test is not None:
        y_pred_test = clf.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        print(f"   Test Accuracy:  {test_acc:.4f}")

        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT - TEST SET")
        print("=" * 70)
        print(classification_report(y_test, y_pred_test, digits=4))

    # Save model
    print("\n5️⃣ Saving model...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"   ✓ Saved model to {output_path}")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()