# 🚀 Arabic NLP Full-Stack Application

A complete full-stack web application for Arabic text classification using Machine Learning and Natural Language Processing.

## 📦 What I've Created

### **Frontend (Web Interface)**
✅ Modern, responsive Arabic RTL interface
✅ Single text prediction with confidence scores
✅ Batch file processing (CSV/TXT)
✅ Real-time statistics and visualizations
✅ Prediction history with localStorage
✅ Beautiful gradient UI with animations

### **Backend (Flask API)**
✅ RESTful API with multiple endpoints
✅ Real-time text classification
✅ Batch prediction support
✅ Arabic text preprocessing
✅ Model loading and management
✅ CORS enabled for frontend integration

### **Preprocessing Modules**
✅ `ArabicCleaner` - Complete text cleaning
✅ `ArabicTokenizer` - Advanced tokenization
✅ Stop words removal
✅ N-grams generation
✅ Text normalization

## 🏗️ Architecture

```
┌─────────────────┐
│   Frontend      │  (HTML/CSS/JavaScript)
│   index.html    │  - User Interface
└────────┬────────┘  - Visualization
         │           - History Management
         │ HTTP/JSON
         ▼
┌─────────────────┐
│   Flask API     │  (Python)
│   app.py        │  - Request handling
└────────┬────────┘  - Model inference
         │           - Response formatting
         │
         ▼
┌─────────────────┐
│  Preprocessing  │  (Python Modules)
│  arabic_cleaner │  - Text cleaning
│  tokenizer      │  - Tokenization
└────────┬────────┘  - Feature extraction
         │
         ▼
┌─────────────────┐
│  ML Model       │  (scikit-learn)
│  best_model.pkl │  - Classification
│  vectorizer.pkl │  - Prediction
└─────────────────┘  - Probabilities
```

## 📁 Complete File Structure

```
arabic_nlp_project/
│
├── 🌐 Frontend
│   └── frontend/
│       └── index.html                    # Web interface
│
├── 🔧 Backend
│   ├── app.py                            # Flask server
│   └── requirements_backend.txt          # Dependencies
│
├── 📚 Source Code
│   └── src/
│       ├── __init__.py
│       └── preprocessing/
│           ├── __init__.py
│           ├── arabic_cleaner.py         # Text cleaning
│           └── tokenizer.py              # Tokenization
│
├── 🤖 Models & Results
│   └── results/
│       ├── models/
│       │   ├── best_model.pkl            # Trained model
│       │   ├── vectorizer.pkl            # Feature extractor
│       │   └── model_results.json        # Metadata
│       ├── figures/                      # Visualizations
│       └── reports/                      # Evaluation reports
│
├── 💾 Data
│   └── data/
│       ├── processed/
│       │   ├── X_train_features.npy
│       │   ├── X_test_features.npy
│       │   ├── y_train.npy
│       │   └── y_test.npy
│       └── raw/
│           └── arabic_tweets.csv
│
└── 📖 Documentation
    ├── SETUP_GUIDE.md                    # Setup instructions
    └── FULLSTACK_README.md               # This file
```

## 🚀 Quick Start

### 1️⃣ Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements_backend.txt
```

### 2️⃣ Train Model (if needed)

```bash
python scripts/train_classifier.py
```

### 3️⃣ Start Backend Server

```bash
python app.py
```

Server runs at: `http://localhost:5000`

### 4️⃣ Open Frontend

Navigate to: `http://localhost:5000`

## 🔌 API Endpoints

### Health Check
```http
GET /api/health
```

### Model Information
```http
GET /api/model-info

Response:
{
  "model_name": "Logistic Regression",
  "accuracy": 0.945,
  "f1_score": 0.942,
  "num_classes": 3
}
```

### Single Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "text": "هذا نص عربي للتصنيف"
}

Response:
{
  "success": true,
  "predicted_class": 0,
  "confidence": 0.92,
  "probabilities": [0.92, 0.05, 0.03],
  "processing_time": 45
}
```

### Batch Prediction
```http
POST /api/predict-batch
Content-Type: application/json

{
  "texts": ["نص 1", "نص 2", "نص 3"]
}

Response:
{
  "success": true,
  "total": 3,
  "successful": 3,
  "failed": 0,
  "processing_time": 120,
  "results": [...]
}
```

### Class Names
```http
GET /api/class-names

Response:
{
  "0": "إيجابي",
  "1": "محايد",
  "2": "سلبي"
}
```

## 💡 Features

### Frontend Features

1. **التنبؤ (Prediction)**
   - Real-time text classification
   - Confidence visualization
   - Processing time display
   - Model information panel

2. **معالجة دفعة (Batch Processing)**
   - CSV/TXT file upload
   - Batch predictions
   - Results download
   - Progress tracking

3. **الإحصائيات (Statistics)**
   - Total predictions counter
   - Average confidence
   - Processing time metrics
   - Class distribution charts

4. **السجل (History)**
   - Last 50 predictions stored
   - Timestamps and metadata
   - Local storage persistence
   - Clear history option

5. **حول النظام (About)**
   - Project information
   - Technical details
   - Performance metrics

### Backend Features

1. **Preprocessing Pipeline**
   - Diacritics removal
   - URL/email cleaning
   - Text normalization
   - Tokenization
   - Stop words removal

2. **Model Management**
   - Auto-load trained model
   - Vectorizer integration
   - Metadata handling
   - Error recovery

3. **API Features**
   - CORS enabled
   - JSON responses
   - Error handling
   - Request validation

## 🧪 Testing

### Test Single Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "هذا اختبار للنظام"}'
```

### Test with Python
```python
import requests

response = requests.post(
    'http://localhost:5000/api/predict',
    json={'text': 'نص عربي للاختبار'}
)
print(response.json())
```

## 🎨 Customization

### Change Class Names

Edit `app.py`:
```python
@app.route('/api/class-names', methods=['GET'])
def get_class_names():
    class_names = {
        0: 'فئة مخصصة 1',
        1: 'فئة مخصصة 2',
        2: 'فئة مخصصة 3'
    }
    return jsonify(class_names)
```

### Adjust Preprocessing

Edit `src/preprocessing/arabic_cleaner.py`:
```python
def clean(self, text):
    # Customize cleaning options
    return self.heavy_clean(text)
```

### Change Model

Replace `results/models/best_model.pkl` with your model.

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python scripts/train_classifier.py` |
| CORS errors | Install `flask-cors`: `pip install flask-cors` |
| Port in use | Change port in `app.py`: `app.run(port=8000)` |
| Import errors | Add to PYTHONPATH: `export PYTHONPATH="${PWD}"` |

## 📊 Performance

- **Prediction Speed**: < 100ms per text
- **Batch Processing**: Up to 1000 texts
- **Accuracy**: 94.5%
- **F1-Score**: 0.942

## 🔒 Security Notes

For production deployment:

1. ✅ Enable HTTPS
2. ✅ Add authentication
3. ✅ Implement rate limiting
4. ✅ Use environment variables
5. ✅ Add input validation
6. ✅ Enable logging

## 🚀 Production Deployment

### Using Gunicorn
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker
```bash
docker build -t arabic-nlp .
docker run -p 5000:5000 arabic-nlp
```

## 📝 Dependencies

**Backend:**
- Flask 2.3.3
- flask-cors 4.0.0
- scikit-learn 1.3.0
- numpy 1.24.3
- pandas 2.0.3

**Frontend:**
- Vanilla JavaScript (no dependencies)
- Modern browsers with ES6+ support

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 👥 Team

Developed as part of Arabic NLP research project.

## 📧 Contact

For questions or support, please create an issue in the repository.

---

**Made with ❤️ for Arabic NLP**