"""
Flask Backend for Arabic NLP Text Classification
Usage: python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import json
from pathlib import Path
import time
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing.arabic_cleaner import ArabicCleaner
from preprocessing.tokenizer import ArabicTokenizer

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)  # Enable CORS for frontend

# Global variables
model = None
vectorizer = None
model_metadata = None
cleaner = None
tokenizer = None

def load_model_and_resources():
    """Load model and preprocessing resources."""
    global model, vectorizer, model_metadata, cleaner, tokenizer
    
    print("Loading model and resources...")
    
    try:
        # Load best model
        model_path = Path('results/models/best_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {model_path}")
        
        # Load vectorizer/feature extractor if exists
        vectorizer_path = Path('results/models/vectorizer.pkl')
        if vectorizer_path.exists():
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"✓ Vectorizer loaded: {vectorizer_path}")
        
        # Load model metadata
        metadata_path = Path('results/models/model_results.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            model_metadata = json.load(f)
        print(f"✓ Metadata loaded: {metadata_path}")
        
        # Initialize preprocessing tools
        cleaner = ArabicCleaner()
        tokenizer = ArabicTokenizer()
        print("✓ Preprocessing tools initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading resources: {e}")
        return False


def preprocess_text(text):
    """Preprocess Arabic text."""
    try:
        # Clean text
        cleaned = cleaner.clean(text)
        
        # Tokenize
        tokens = tokenizer.tokenize(cleaned)
        
        # Rejoin for vectorization
        processed = ' '.join(tokens)
        
        return processed
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return text


@app.route('/')
def index():
    """Serve the main HTML page."""
    return send_from_directory('frontend', 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get model information."""
    if not model_metadata:
        return jsonify({'error': 'Model metadata not loaded'}), 500
    
    return jsonify({
        'model_name': model_metadata.get('best_model', 'Unknown'),
        'accuracy': model_metadata.get('best_accuracy', 0),
        'f1_score': model_metadata.get('best_f1_score', 0),
        'num_classes': len(model_metadata.get('all_models', {}).get(model_metadata.get('best_model', ''), {}).get('report', {}).keys()) if 'all_models' in model_metadata else 3
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict classification for a single text."""
    start_time = time.time()
    
    try:
        # Get text from request
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize if vectorizer exists
        if vectorizer:
            features = vectorizer.transform([processed_text])
        else:
            # If no vectorizer, assume model handles raw text
            features = [processed_text]
        
        # Make prediction
        prediction = model.predict(features)
        predicted_class = int(prediction[0])
        
        # Get confidence if available
        confidence = 0.0
        probabilities = []
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            probabilities = proba[0].tolist()
            confidence = float(proba[0].max())
        else:
            # Default confidence for models without probability
            confidence = 0.85
            probabilities = [0.0] * 3
            probabilities[predicted_class] = confidence
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'processing_time': processing_time,
            'original_text': text[:100],
            'processed_text': processed_text[:100]
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Predict classification for multiple texts."""
    start_time = time.time()
    
    try:
        # Get texts from request
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({'error': 'No texts provided'}), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({'error': 'Texts must be a list'}), 400
        
        if len(texts) == 0:
            return jsonify({'error': 'Empty texts list'}), 400
        
        # Limit batch size
        if len(texts) > 1000:
            return jsonify({'error': 'Batch size too large (max 1000)'}), 400
        
        results = []
        successful = 0
        failed = 0
        
        for idx, text in enumerate(texts):
            try:
                # Preprocess
                processed_text = preprocess_text(text)
                
                # Vectorize
                if vectorizer:
                    features = vectorizer.transform([processed_text])
                else:
                    features = [processed_text]
                
                # Predict
                prediction = model.predict(features)
                predicted_class = int(prediction[0])
                
                # Get confidence
                confidence = 0.0
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    confidence = float(proba[0].max())
                else:
                    confidence = 0.85
                
                results.append({
                    'index': idx,
                    'text': text[:100],
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'status': 'success'
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'text': text[:100],
                    'error': str(e),
                    'status': 'failed'
                })
                failed += 1
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)  # ms
        
        return jsonify({
            'success': True,
            'total': len(texts),
            'successful': successful,
            'failed': failed,
            'processing_time': processing_time,
            'results': results
        })
        
    except Exception as e:
        print(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/upload-file', methods=['POST'])
def upload_file():
    """Handle file upload for batch processing."""
    try:
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        # Read file content
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Assume first column or 'text' column contains texts
            if 'text' in df.columns:
                texts = df['text'].tolist()
            else:
                texts = df.iloc[:, 0].tolist()
                
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        return jsonify({
            'success': True,
            'filename': file.filename,
            'num_texts': len(texts),
            'texts': texts[:5]  # Return first 5 for preview
        })
        
    except Exception as e:
        print(f"File upload error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get model statistics."""
    try:
        stats_path = Path('results/reports/production_results.json')
        
        if not stats_path.exists():
            return jsonify({
                'error': 'Statistics not available'
            }), 404
        
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        
        return jsonify(stats)
        
    except Exception as e:
        print(f"Statistics error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/class-names', methods=['GET'])
def get_class_names():
    """Get class names mapping."""
    # Default class names (customize based on your dataset)
    class_names = {
        0: 'إيجابي',
        1: 'محايد',
        2: 'سلبي'
    }
    
    return jsonify(class_names)


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ARABIC NLP TEXT CLASSIFICATION - BACKEND SERVER")
    print("="*80)
    
    # Load model and resources
    success = load_model_and_resources()
    
    if not success:
        print("\n⚠️  WARNING: Some resources could not be loaded!")
        print("The server will start but predictions may not work correctly.")
    
    print("\n" + "="*80)
    print("Starting Flask server...")
    print("="*80)
    print("\n✓ Server running at: http://localhost:5000")
    print("✓ API endpoints available at: http://localhost:5000/api/*")
    print("✓ Frontend available at: http://localhost:5000/")
    print("\nPress CTRL+C to stop the server\n")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )