"""
Run predictions on new data.
Usage: python scripts/predict.py --model results/models/best_model.pkl --features features.npy --output predictions.csv
"""

import argparse
import numpy as np
import pandas as pd
from joblib import load

def main(model_path, features_path, output_path):
    model = load(model_path)
    X = np.load(features_path)
    preds = model.predict(X)
    pd.DataFrame({'prediction': preds}).to_csv(output_path, index=False)
    print("Predictions saved to", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args.model, args.features, args.output)