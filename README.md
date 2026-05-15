# AyurMitraAI - Ayurvedic Intelligence Platform

## Project Title
AyurMitraAI: AI-Powered Ayurvedic Disease Prediction and Recommendation System

## Abstract
AyurMitraAI is a comprehensive web-based platform that leverages deep learning and traditional Ayurvedic knowledge to provide personalized health insights. The system uses a hybrid ANN+LSTM neural network to analyze user symptoms and lifestyle factors, predict potential Ayurvedic imbalances, and provide tailored recommendations including herbs, diet, yoga, and preventive measures. Built with Flask and TensorFlow, it combines modern machine learning with ancient Ayurvedic wisdom for holistic wellness guidance.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ayurmedha.git
cd ayurmedha
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model (optional, pre-trained artifacts included):
```bash
python train_model.py
```

5. Run the application:
```bash
python app.py
```

6. Access the application at: http://localhost:5000

### Database Setup
The application uses SQLite database which is created automatically on first run.

## 1. Overall System Flow

```
User Input (Symptoms + Lifestyle)
        ↓
Preprocessing (Feature Engineering)
        ↓
Deep Learning Model (ANN+LSTM)
        ↓
Disease Prediction with Confidence
        ↓
Ayurvedic Knowledge Base Lookup
        ↓
Personalized Recommendations
        ↓
Result Display with Explanations
```

## 2. Deep Learning Model Implementation

### Model Architecture
- **Input Layer**: Structured features (19 dimensions) + Text sequences (50 tokens)
- **ANN Branch**: Dense layers for structured data (256 → 128 → 64 neurons)
- **LSTM Branch**: Bidirectional LSTM for symptom text processing (64 units)
- **Fusion Layer**: Concatenation and final dense layers (128 → 64 → n_classes)
- **Output**: Softmax probabilities for disease classification

### Training Details
- **Dataset**: AyurGenixAI Dataset (446 samples, 34+ features)
- **Preprocessing**: Label encoding, feature scaling, text tokenization
- **Balancing**: SMOTE for class imbalance
- **Optimization**: Adam optimizer, categorical cross-entropy loss
- **Regularization**: L2 regularization, dropout, batch normalization
- **Callbacks**: Early stopping, learning rate reduction

### Model Artifacts
- `model.keras`: Trained TensorFlow model
- `scaler.pkl`: Feature scaler
- `tokenizer.pkl`: Text tokenizer
- `label_encoder.pkl`: Target encoder
- `label_encoders.pkl`: Categorical feature encoders

## 3. Ayurvedic Knowledge Integration

### Knowledge Base
- **Source**: AyurGenixAI Dataset with comprehensive Ayurvedic mappings
- **Coverage**: Doshas, herbs, formulations, diet, lifestyle, yoga, prognosis
- **Fallback**: Similarity-based recommendations using TF-IDF

### Integration Approach
- **Rule-Based**: Direct mapping from predicted disease to Ayurvedic recommendations
- **Similarity Matching**: Cosine similarity for unknown conditions
- **Confidence Threshold**: 80% threshold for specific recommendations vs. general wellness

## 4. Step-by-Step Pipeline

1. **User Registration/Login**: Secure user authentication
2. **Symptom Input**: Free-text symptoms + lifestyle parameters
3. **Preprocessing**:
   - Text cleaning and tokenization
   - Feature extraction (symptom severity, duration)
   - Categorical encoding and scaling
4. **Model Inference**:
   - ANN+LSTM prediction
   - Confidence scoring
   - Threshold application (80%)
5. **Recommendation Generation**:
   - Dataset lookup for predicted disease
   - Ayurvedic mapping extraction
   - Fallback handling
6. **Explainable AI**:
   - SHAP value computation
   - Dynamic explanation generation
   - Feature importance analysis
7. **Result Presentation**:
   - Disease prediction (if confident)
   - Ayurvedic recommendations
   - Alternative predictions
   - Interactive explanations

## 5. Why This Is Genuinely Deep Learning

### Neural Network Components
- **Multi-Input Architecture**: Separate processing for structured and text data
- **Deep Layers**: Multiple hidden layers with non-linear transformations
- **Sequence Modeling**: LSTM for temporal symptom patterns
- **Feature Learning**: Automatic feature extraction from raw inputs

### Advanced Techniques
- **Hybrid Model**: ANN + LSTM combination for multimodal learning
- **Regularization**: L2, dropout, batch norm for generalization
- **Balancing**: SMOTE for handling imbalanced medical data
- **Explainability**: SHAP integration for model interpretability

### Scalability and Performance
- **Large Vocabulary**: 10,000 word tokenizer
- **Efficient Training**: Early stopping and adaptive learning rates
- **Production Ready**: Model serialization and inference optimization

## Dataset Information
- **Primary Dataset**: AyurGenixAI_Dataset.csv (446 records)
- **Features**: 34+ columns including symptoms, demographics, medical history
- **Target**: Ayurvedic disease classifications
- **Preprocessing**: Feature selection, encoding, normalization

## Ethical Considerations
- **No Medical Diagnosis**: Clear disclaimers about not replacing professional medical advice
- **Ayurvedic Focus**: Recommendations based on traditional knowledge
- **Confidence Thresholds**: Conservative prediction requirements
- **User Privacy**: Secure data handling and storage

## Technology Stack
- **Backend**: Flask, SQLAlchemy, TensorFlow
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite
- **ML Libraries**: scikit-learn, Keras, SHAP
- **Deployment**: Local development server

## Future Enhancements
- Real-time health monitoring integration
- Expanded Ayurvedic knowledge base
- Multi-language support
- Mobile application development
- Advanced visualization dashboards

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
This project is for educational and research purposes. Please consult healthcare professionals for medical advice.