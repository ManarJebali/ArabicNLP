ARABIC NLP PIPELINE
1️⃣ Loading data...
✓ Loaded 15000 samples
2️⃣ Cleaning text...
✓ Cleaned 15000 texts
3️⃣ Splitting data...
✓ Train: 12000, Test: 3000
4️⃣ Tokenizing...
Vocabulary size: 5000
✓ Vocabulary built
5️⃣ Training embeddings...
Epoch 1/5 - avg loss: 2.8849
...
✓ Embeddings: (5001, 50)
6️⃣ Extracting features...
✓ Features: (12000, 50)
7️⃣ Training classifier...
✓ Classifier trained
8️⃣ Evaluating...
Train Accuracy: 0.9234
Test Accuracy: 0.8567
==================================================================
✅ Pipeline Complete!

## Troubleshooting

See README.md for common issues and solutions.

Quick Command Reference
bash# Full pipeline
python scripts/run_pipeline.py --data data/raw/your_data.xlsx

# Train embeddings only
python scripts/train_embeddings.py --data data/raw/your_data.xlsx

# Run tests
pytest tests/

# Launch Jupyter
jupyter notebook notebooks/
