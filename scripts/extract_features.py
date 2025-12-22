"""
Script to extract features from tokens.json using the feature pipeline.
Usage: python scripts/extract_features.py --data data/processed/train.csv --output features.npy --config configs/default_config.yaml
"""

import argparse
import pandas as pd
import yaml
import numpy as np

from src.utils.data_loader import load_arabic_dataset
from src.preprocessing.tokenizer import tokenize_arabic
from src.features.pipeline import ArabicFeaturePipeline

def main(data_path, output_path, config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(data_path)
    df['tokens.json'] = df['text'].apply(tokenize_arabic)

    pipeline = ArabicFeaturePipeline(config=config['features'])
    features = pipeline.transform(df['tokens.json'])
    np.save(output_path, features)
    print("Features extracted and saved to", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.data, args.output, args.config)
