"""
Run predictions on new data
Usage: python scripts/predict.py --model results/models/best_model.pkl --features data/processed/X_test_features.npy --output results/predictions.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description='Run Predictions')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model (pkl)')
    parser.add_argument('--features', type=str, required=True,
                        help='Path to features (npy)')
    parser.add_argument('--labels', type=str, default=None,
                        help='Path to true labels (npy) - optional for evaluation')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for predictions (csv)')
    args = parser.parse_args()

    print("=" * 70)
    print("RUNNING PREDICTIONS")
    print("=" * 70)

    # Load model
    print("\n1️⃣ Loading model...")
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    print(f"   ✓ Model loaded from {args.model}")

    # Load features
    print("\n2️⃣ Loading features...")
    X = np.load(args.features)
    print(f"   ✓ Features shape: {X.shape}")

    # Load labels if provided
    if args.labels:
        y_true = np.load(args.labels)
        print(f"   ✓ Labels loaded: {y_true.shape}")
    else:
        y_true = None

    # Predict
    print("\n3️⃣ Making predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
    print(f"   ✓ Generated {len(predictions)} predictions")

    # Evaluate if labels provided
    if y_true is not None:
        from sklearn.metrics import accuracy_score, classification_report
        acc = accuracy_score(y_true, predictions)
        print(f"\n   Accuracy: {acc:.4f}")
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT")
        print("=" * 70)
        print(classification_report(y_true, predictions, digits=4))

    # Save predictions
    print("\n4️⃣ Saving predictions...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = pd.DataFrame({'prediction': predictions})

    if probabilities is not None:
        for i in range(probabilities.shape[1]):
            result_df[f'prob_class_{i}'] = probabilities[:, i]

    if y_true is not None:
        result_df['true_label'] = y_true

    result_df.to_csv(output_path, index=False)
    print(f"   ✓ Predictions saved to {output_path}")

    print("\n" + "=" * 70)
    print("✅ PREDICTIONS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()