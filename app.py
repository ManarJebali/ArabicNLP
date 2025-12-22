"""
Fixed Flask Backend for Arabic NLP Text Classification
Place in: C:\Users\PCS\Desktop\ArabicNLP-master\backend\app.py
Run from: C:\Users\PCS\Desktop\ArabicNLP-master\
Command: python backend\app.py
"""

import re
import json
import time
import pickle
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

print("="*80)
print("ARABIC NLP BACKEND - STARTING")
print("="*80)

# Get absolute paths
SCRIPT_DIR = Path(__file__).resolve().parent  # backend/
ROOT_DIR = SCRIPT_DIR.parent  # ArabicNLP-master/
MODELS_DIR = ROOT_DIR / "results" / "models"
FRONTEND_DIR = SCRIPT_DIR / "frontend"

print(f"\nScript directory: {SCRIPT_DIR}")
print(f"Root directory: {ROOT_DIR}")
print(f"Models directory: {MODELS_DIR}")
print(f"Frontend directory: {FRONTEND_DIR}")
print(f"Models exists: {MODELS_DIR.exists()}")
print(f"Frontend exists: {FRONTEND_DIR.exists()}")

# Simple Arabic text cleaner
class SimpleArabicCleaner:
    def __init__(self):
        self.arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
        self.url_pattern = re.compile(r'http[s]?://\S+')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.spaces_pattern = re.compile(r'\s+')
    
    def clean(self, text):
        if not text:
            return ""
        text = self.arabic_diacritics.sub('', text)
        text = self.url_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        text = self.spaces_pattern.sub(' ', text).strip()
        return text

# Global variables
model = None
vectorizer = None
model_metadata = None
class_names_map = {}
cleaner = SimpleArabicCleaner()

def load_resources():
    """Load model, vectorizer, and metadata"""
    global model, vectorizer, model_metadata, class_names_map
    
    print("\n" + "="*80)
    print("LOADING RESOURCES")
    print("="*80)
    
    try:
        # Check models directory
        if not MODELS_DIR.exists():
            print(f"\n✗ Models directory not found: {MODELS_DIR}")
            print("\nPlease train the model first:")
            print("  python train_model_final.py")
            return False
        
        # Load model
        model_path = MODELS_DIR / 'best_model.pkl'
        if not model_path.exists():
            print(f"\n✗ Model file not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {type(model).__name__}")
        
        # Load vectorizer
        vectorizer_path = MODELS_DIR / 'vectorizer.pkl'
        if vectorizer_path.exists():
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"✓ Vectorizer loaded: {type(vectorizer).__name__}")
        else:
            print("⚠ Vectorizer not found")
        
        # Load metadata
        metadata_path = MODELS_DIR / 'model_results.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
            print("✓ Metadata loaded")
        else:
            model_metadata = {
                'best_model': 'Model',
                'best_accuracy': 0.0,
                'best_f1_score': 0.0
            }
        
        # Load class names
        class_names_path = MODELS_DIR / 'class_names.json'
        if class_names_path.exists():
            with open(class_names_path, 'r', encoding='utf-8') as f:
                class_names_map = json.load(f)
            print("✓ Class names loaded")
        else:
            class_names_map = {'0': 'فئة 0', '1': 'فئة 1', '2': 'فئة 2'}
        
        print("\n" + "="*80)
        print("✅ ALL RESOURCES LOADED SUCCESSFULLY")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

# Create Flask app
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Load resources
MODEL_LOADED = load_resources()

@app.route('/')
def index():
    """Serve frontend"""
    try:
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    except Exception as e:
        return jsonify({'error': 'Frontend not found', 'path': str(FRONTEND_DIR)}), 404

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL_LOADED and model is not None,
        'vectorizer_loaded': vectorizer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not MODEL_LOADED or not model_metadata:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_name': model_metadata.get('best_model', 'Unknown'),
        'accuracy': model_metadata.get('best_accuracy', 0),
        'f1_score': model_metadata.get('best_f1_score', 0),
        'num_classes': len(class_names_map)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict single text"""
    start_time = time.time()
    
    if not MODEL_LOADED or model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': 'Empty text'}), 400
        
        # Preprocess
        cleaned = cleaner.clean(text)
        if not cleaned:
            cleaned = text
        
        # Vectorize
        if not vectorizer:
            return jsonify({'success': False, 'error': 'Vectorizer not loaded'}), 500
        
        features = vectorizer.transform([cleaned])
        
        # Predict
        prediction = model.predict(features)
        predicted_class = int(prediction[0])
        
        # Get confidence
        confidence = 0.85
        probabilities = []
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            probabilities = proba[0].tolist()
            confidence = float(max(probabilities))
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'processing_time': processing_time,
            'original_text': text[:100],
            'processed_text': cleaned[:100]
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction"""
    start_time = time.time()
    
    if not MODEL_LOADED or model is None:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'success': False, 'error': 'No texts provided'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'success': False, 'error': 'Invalid texts'}), 400
        
        if len(texts) > 1000:
            return jsonify({'success': False, 'error': 'Too many texts (max 1000)'}), 400
        
        results = []
        successful = 0
        failed = 0
        
        for idx, text in enumerate(texts):
            try:
                cleaned = cleaner.clean(str(text))
                if not cleaned:
                    cleaned = str(text)
                
                features = vectorizer.transform([cleaned])
                prediction = model.predict(features)
                predicted_class = int(prediction[0])
                
                confidence = 0.85
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)
                    confidence = float(proba[0].max())
                
                results.append({
                    'index': idx,
                    'text': str(text)[:100],
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'status': 'success'
                })
                successful += 1
                
            except Exception as e:
                results.append({
                    'index': idx,
                    'text': str(text)[:100],
                    'error': str(e),
                    'status': 'failed'
                })
                failed += 1
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'total': len(texts),
            'successful': successful,
            'failed': failed,
            'processing_time': processing_time,
            'results': results
        })
        
    except Exception as e:
        print(f"Batch error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/class-names', methods=['GET'])
def get_class_names():
    """Get class names mapping"""
    return jsonify(class_names_map)

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING FLASK SERVER")
    print("="*80)
    
    if not MODEL_LOADED:
        print("\n⚠️  WARNING: Model not loaded!")
        print("Server will start but predictions won't work.")
        print("\nTo fix:")
        print("  1. python train_model_final.py")
        print("  2. Wait for completion")
        print("  3. Restart server")
    
    print("\n✓ Server at: http://localhost:5000")
    print("✓ Frontend: http://localhost:5000")
    print("✓ Health: http://localhost:5000/api/health")
    print("\nPress CTRL+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )
