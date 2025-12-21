# Arabic NLP Project - Setup Guide

## 📋 Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Clone and Setup Environment

```bash
# Navigate to project directory
cd arabic_nlp_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements_backend.txt
```

### 2. Train Your Model (if not already trained)

```bash
# Run the complete training pipeline
python scripts/train_classifier.py

# Or run individual steps:
python scripts/train_embeddings.py
python scripts/extract_features.py
python scripts/train_classifier.py
```

### 3. Setup Frontend

```bash
# Create frontend directory
mkdir frontend

# Move the HTML file to frontend directory
mv index.html frontend/

# The Flask app will serve it automatically
```

### 4. Start the Backend Server

```bash
# Run Flask server
python app.py
```

The server will start at: `http://localhost:5000`

### 5. Access the Application

Open your browser and navigate to:
```
http://localhost:5000
```

## 📁 Required Project Structure

```
arabic_nlp_project/
├── app.py                           # Flask backend server
├── requirements_backend.txt         # Backend dependencies
├── frontend/
│   └── index.html                   # Web interface
├── src/
│   ├── preprocessing/
│   │   ├── arabic_cleaner.py
│   │   └── tokenizer.py
│   └── ...
├── data/
│   └── processed/
│       ├── X_train_features.npy
│       ├── X_test_features.npy
│       ├── y_train.npy
│       └── y_test.npy
└── results/
    └── models/
        ├── best_model.pkl
        ├── vectorizer.pkl (optional)
        └── model_results.json
```

## 🔧 API Endpoints

### Health Check
```
GET http://localhost:5000/api/health
```

### Model Information
```
GET http://localhost:5000/api/model-info
```

### Single Prediction
```
POST http://localhost:5000/api/predict
Content-Type: application/json

{
  "text": "نص عربي للتصنيف"
}
```

### Batch Prediction
```
POST http://localhost:5000/api/predict-batch
Content-Type: application/json

{
  "texts": ["نص 1", "نص 2", "نص 3"]
}
```

### Get Class Names
```
GET http://localhost:5000/api/class-names
```

## 🔍 Testing the API

### Using cURL:

```bash
# Test health endpoint
curl http://localhost:5000/api/health

# Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "هذا نص تجريبي للتصنيف"}'
```

### Using Python:

```python
import requests

# Single prediction
response = requests.post(
    'http://localhost:5000/api/predict',
    json={'text': 'نص عربي للتصنيف'}
)
print(response.json())

# Batch prediction
response = requests.post(
    'http://localhost:5000/api/predict-batch',
    json={'texts': ['نص 1', 'نص 2', 'نص 3']}
)
print(response.json())
```

## 🐛 Troubleshooting

### Issue: Model not found
**Solution:** Make sure you've trained the model first:
```bash
python scripts/train_classifier.py
```

### Issue: CORS errors in browser
**Solution:** The Flask app already has CORS enabled via `flask-cors`. Make sure it's installed:
```bash
pip install flask-cors
```

### Issue: Preprocessing modules not found
**Solution:** Make sure the `src` directory is in your Python path. The `app.py` already adds it, but you can also:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
```

### Issue: Port 5000 already in use
**Solution:** Change the port in `app.py`:
```python
app.run(host='0.0.0.0', port=8000, debug=True)
```

## 🚀 Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements_backend.txt .
RUN pip install -r requirements_backend.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t arabic-nlp .
docker run -p 5000:5000 arabic-nlp
```

## 📊 Customization

### Change Class Names

Edit the `get_class_names()` function in `app.py`:

```python
@app.route('/api/class-names', methods=['GET'])
def get_class_names():
    class_names = {
        0: 'فئة أولى',
        1: 'فئة ثانية',
        2: 'فئة ثالثة'
    }
    return jsonify(class_names)
```

### Add Authentication

Install Flask-Login:
```bash
pip install flask-login
```

Add to `app.py`:
```python
from flask_login import LoginManager, login_required

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    # Your code here
```

## 📝 Notes

- The frontend uses localStorage to save prediction history
- Statistics are calculated client-side
- For production, consider using a proper database
- Enable HTTPS for production deployment
- Add rate limiting for production use

## 🤝 Support

For issues or questions, please create an issue in the project repository.