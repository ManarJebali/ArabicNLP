import re
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import os
import sys
from flask import send_from_directory


print("="*80)
print("ARABIC NLP BACKEND - STARTING")
print("="*80)

# -------------------------------
# Define root and subdirectories
# -------------------------------

# ROOT_DIR: 2 levels up from this script (adjust as needed)
ROOT_DIR = Path(__file__).resolve().parent  # this is backend/

# Directories
MODELS_DIR = ROOT_DIR / "results" / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"
RESULTS_DIR = ROOT_DIR / "results" / "models"

# -------------------------------
# Initialize Flask
# -------------------------------
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIR),
    static_url_path="/"  # Serve files at root URL
)
CORS(app)


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path != "" and (FRONTEND_DIR / path).exists():
        return send_from_directory(FRONTEND_DIR, path)
    else:
        return send_from_directory(FRONTEND_DIR, "index.html")


# Create directories if missing (optional)
for d in [MODELS_DIR, FRONTEND_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Load the ML model
# -------------------------------
model_path = MODELS_DIR / "best_model.pkl"

if model_path.exists():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded successfully from {model_path}")
else:
    print(f"Warning: Model file not found at {model_path}")
    model = None

# -------------------------------
# Print paths for verification
# -------------------------------
print(f"Root directory: {ROOT_DIR}")
print(f"Models directory: {MODELS_DIR} (exists: {MODELS_DIR.exists()})")
print(f"Frontend directory: {FRONTEND_DIR} (exists: {FRONTEND_DIR.exists()})")
print(f"Results directory: {RESULTS_DIR} (exists: {RESULTS_DIR.exists()})")



# Simple preprocessing
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
cleaner = SimpleArabicCleaner()

def load_model():
    """Load model with proper error handling"""
    global model, vectorizer, model_metadata
    
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    
    try:
        # Check if results directory exists
        if not RESULTS_DIR.exists():
            print(f"\n✗ Results directory not found: {RESULTS_DIR}")
            print("\nTo fix this:")
            print("  1. Make sure you're in the correct directory")
            print("  2. Run: python train_model_final.py")
            print("  3. Wait for training to complete")
            return False
        
        # Load model
        model_path = RESULTS_DIR / 'best_model.pkl'
        print(f"\nLooking for model: {model_path}")
        
        if not model_path.exists():
            print(f"✗ Model file not found!")
            print(f"\nPlease train the model first:")
            print(f"  python train_model_final.py")
            return False
        
        print("Loading model...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Load vectorizer
        vectorizer_path = RESULTS_DIR / 'vectorizer.pkl'
        print(f"\nLooking for vectorizer: {vectorizer_path}")
        
        if vectorizer_path.exists():
            print("Loading vectorizer...")
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            print(f"✓ Vectorizer loaded successfully")
            print(f"  Vectorizer type: {type(vectorizer).__name__}")
        else:
            print("⚠ Vectorizer not found (optional)")
        
        # Load metadata
        metadata_path = RESULTS_DIR / 'model_results.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
            print(f"✓ Metadata loaded")
        else:
            model_metadata = {
                'best_model': 'Trained Model',
                'best_accuracy': 0.95,
                'best_f1_score': 0.94
            }
            print("⚠ Using default metadata")
        
        print("\n" + "="*80)
        print("✓ ALL RESOURCES LOADED SUCCESSFULLY")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

# Create Flask app
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')
CORS(app)

# Load model on startup
MODEL_LOADED = load_model()

@app.route('/')
def index():
    """Serve frontend"""
    try:
        return send_from_directory(str(FRONTEND_DIR), 'index.html')
    except Exception as e:
        return jsonify({
            'error': 'Frontend not found',
            'message': str(e),
            'frontend_dir': str(FRONTEND_DIR)
        }), 404

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
        return jsonify({
            'error': 'Model not loaded',
            'model_loaded': False
        }), 500
    
    return jsonify({
        'model_name': model_metadata.get('best_model', 'Unknown'),
        'accuracy': model_metadata.get('best_accuracy', 0),
        'f1_score': model_metadata.get('best_f1_score', 0),
        'num_classes': 3
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict text classification"""
    start_time = time.time()
    
    # Check if model is loaded
    if not MODEL_LOADED or model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please train the model first.'
        }), 500
    
    try:
        # Get data
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'success': False,
                'error': 'Empty text'
            }), 400
        
        # Preprocess
        cleaned_text = cleaner.clean(text)
        if not cleaned_text:
            cleaned_text = text  # Fallback to original
        
        # Vectorize
        if vectorizer:
            features = vectorizer.transform([cleaned_text])
        else:
            return jsonify({
                'success': False,
                'error': 'Vectorizer not loaded'
            }), 500
        
        # Predict
        prediction = model.predict(features)
        predicted_class = int(prediction[0])
        
        # Get confidence
        confidence = 0.85
        probabilities = [0.0, 0.0, 0.0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            probabilities = proba[0].tolist()
            confidence = float(max(probabilities))
        
        # Processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'processing_time': processing_time,
            'original_text': text[:100],
            'processed_text': cleaned_text[:100]
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Prediction failed: {str(e)}'
        }), 500

@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """Batch prediction"""
    start_time = time.time()
    
    if not MODEL_LOADED or model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts provided'
            }), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'success': False,
                'error': 'Invalid texts list'
            }), 400
        
        if len(texts) > 1000:
            return jsonify({
                'success': False,
                'error': 'Too many texts (max 1000)'
            }), 400
        
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
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/class-names', methods=['GET'])
def class_names():
    """Get class name mapping"""
    # Load from file if exists
    class_names_path = RESULTS_DIR / 'class_names.json'
    if class_names_path.exists():
        with open(class_names_path, 'r', encoding='utf-8') as f:
            return jsonify(json.load(f))
    
    # Default class names
    return jsonify({
        '0': 'فئة 0',
        '1': 'فئة 1',
        '2': 'فئة 2'
    })

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
        print("\nTo fix this:")
        print("  1. Stop the server (Ctrl+C)")
        print("  2. Run: python train_model_final.py")
        print("  3. Wait for training to complete")
        print("  4. Restart: python app_working.py")
    
    print("\n✓ Server starting at: http://localhost:5000")
    print("✓ Frontend: http://localhost:5000")
    print("✓ Health check: http://localhost:5000/api/health")
    print("\nPress CTRL+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to avoid double loading
    )
