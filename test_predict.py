"""
test_predict.py - Standalone prediction tester
Run this to test if your model is working correctly
Usage: python test_predict.py
"""

import numpy as np
import pandas as pd
import pickle
import os
import re
import sys

# ML imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==========================
# CONFIG
# ==========================

MAX_LEN = 50
DATA_PATH = "dataset/AyurGenixAI_Dataset.csv"

# Paths to artifacts
MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
LABEL_ENCODERS_PATH = "label_encoders.pkl"

# ==========================
# HELPER FUNCTIONS (Copy from app.py to avoid circular import)
# ==========================

def normalize_symptoms_text(text):
    """Normalize free-text symptoms for stable tokenization."""
    if text is None:
        return ""
    cleaned = str(text).strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s,]", " ", cleaned)
    cleaned = cleaned.replace(",", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def extract_symptom_features(text):
    """Extract numeric features from symptoms text."""
    if pd.isna(text):
        return {'num_symptoms': 0, 'severity': 0, 'duration': 0}
    
    t = str(text).lower()
    symptoms = len(re.split(r'[,;]|\sand\s', t))
    
    severity = 0
    for kw, s in [('mild',1), ('moderate',2), ('severe',3), ('extreme',4)]:
        if kw in t:
            severity = max(severity, s)
    
    duration = 0
    match = re.search(r'(\d+)\s*(day|week|month)', t)
    if match:
        num, unit = int(match.group(1)), match.group(2)
        duration = num * {'day':1, 'week':7, 'month':30}.get(unit, 0)
    
    return {'num_symptoms': min(symptoms,20), 'severity': severity, 'duration': min(duration,365)}

def preprocess_user_input(symptoms, age, weight, sleep, stress, tokenizer, scaler, label_encoders, df):
    """Preprocess user input for prediction (standalone version)"""
    
    if tokenizer is None or scaler is None or label_encoders is None:
        print("ERROR: Artifacts not loaded properly")
        return None, None
    
    if df is None:
        print("ERROR: Dataset not loaded")
        return None, None
    
    # Get all columns except 'Disease'
    all_columns = [col for col in df.columns if col != 'Disease']
    
    # Create user data with ALL columns
    user_data = {col: 'unknown' for col in all_columns}
    user_data['Symptoms'] = symptoms
    
    # Map inputs to columns
    if age > 0:
        age_group_start = int(age // 10 * 10)
        user_data['Age Group'] = f"{age_group_start}-{age_group_start + 9} years"
    user_data['Sleep Patterns'] = f"{int(sleep)} hours" if sleep > 0 else 'unknown'
    
    if stress > 7:
        user_data['Stress Levels'] = 'High'
    elif stress > 4:
        user_data['Stress Levels'] = 'Moderate'
    else:
        user_data['Stress Levels'] = 'Low'
    
    df_user = pd.DataFrame([user_data])
    
    # Clean symptoms
    df_user['Symptoms_clean'] = df_user['Symptoms'].apply(lambda x: normalize_symptoms_text(x))
    
    # Extract numeric features
    symptom_features = df_user['Symptoms'].apply(extract_symptom_features).apply(pd.Series)
    
    # Encode categorical features
    cat_cols = [c for c in all_columns if c != 'Symptoms' and c in label_encoders]
    
    encoded = []
    for col in cat_cols:
        if col in label_encoders:
            series = df_user[col].fillna('unknown').astype(str).str.lower()
            le = label_encoders[col]
            enc = np.array([le.transform([v])[0] if v in le.classes_ else 0 for v in series])
            encoded.append(pd.DataFrame({f'{col}_enc': enc}))
    
    # Combine features
    structured_df = pd.concat([symptom_features] + encoded, axis=1)
    
    # Ensure column order matches scaler
    if hasattr(scaler, 'feature_names_in_'):
        expected_columns = scaler.feature_names_in_
        for col in expected_columns:
            if col not in structured_df.columns:
                structured_df[col] = 0
        structured_df = structured_df[expected_columns]
    
    # Scale
    structured_scaled = scaler.transform(structured_df)
    
    # Tokenize text
    seqs = tokenizer.texts_to_sequences(df_user['Symptoms_clean'].tolist())
    text_padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    
    return structured_scaled.astype(np.float32), text_padded.astype(np.int32)

def get_recommendations(disease_name, df):
    """Get recommendations from dataset"""
    if df is None:
        return {"error": "Dataset not loaded"}
    
    disease_name = str(disease_name).strip().lower()
    
    # Try exact match
    row = df[df["Disease"].str.lower() == disease_name]
    
    # Try partial match
    if row.empty:
        row = df[df["Disease"].str.lower().str.contains(disease_name, na=False)]
    
    if row.empty:
        return {"error": f"Disease '{disease_name}' not found"}
    
    row = row.iloc[0]
    
    recommendations = {
        "doshas": str(row.get("Doshas", ""))[:100] if pd.notna(row.get("Doshas")) else "Not available",
        "herbs": str(row.get("Ayurvedic Herbs", ""))[:100] if pd.notna(row.get("Ayurvedic Herbs")) else "Not available",
        "diet": str(row.get("Diet and Lifestyle Recommendations", ""))[:100] if pd.notna(row.get("Diet and Lifestyle Recommendations")) else "Not available",
        "lifestyle": str(row.get("Yoga & Physical Therapy", ""))[:100] if pd.notna(row.get("Yoga & Physical Therapy")) else "Not available"
    }
    
    return recommendations

# ==========================
# MAIN TEST FUNCTION
# ==========================

def load_artifacts():
    """Load all ML artifacts"""
    print("\n" + "="*60)
    print("Loading ML Artifacts...")
    print("="*60)
    
    model = None
    tokenizer = None
    scaler = None
    label_encoder = None
    label_encoders = None
    df = None
    
    # Load model
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Model error: {e}")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
    
    # Load tokenizer
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded")
    else:
        print(f"❌ Tokenizer not found")
    
    # Load scaler
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
    else:
        print(f"❌ Scaler not found")
    
    # Load label encoder
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder loaded ({len(label_encoder.classes_)} classes)")
    else:
        print(f"❌ Label encoder not found")
    
    # Load feature encoders
    if os.path.exists(LABEL_ENCODERS_PATH):
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        print(f"✅ Feature encoders loaded ({len(label_encoders)} features)")
    else:
        print(f"❌ Feature encoders not found")
    
    # Load dataset
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded ({len(df)} rows)")
    else:
        print(f"❌ Dataset not found at {DATA_PATH}")
    
    print("="*60)
    return model, tokenizer, scaler, label_encoder, label_encoders, df

def run_tests(model, tokenizer, scaler, label_encoder, label_encoders, df):
    """Run prediction tests"""
    
    print("\n" + "="*60)
    print("Running Prediction Tests")
    print("="*60)
    
    # Test cases
    test_cases = [
        {
            "name": "Test 1: Joint Pain",
            "symptoms": "joint pain and fatigue, mild swelling in knees",
            "age": 45, "weight": 75, "sleep": 6, "stress": 5
        },
        {
            "name": "Test 2: Digestive Issues",
            "symptoms": "bloating, indigestion, stomach pain after meals",
            "age": 30, "weight": 65, "sleep": 7, "stress": 6
        },
        {
            "name": "Test 3: Respiratory",
            "symptoms": "cough, cold, sneezing, runny nose for 3 days",
            "age": 25, "weight": 70, "sleep": 5, "stress": 4
        },
        {
            "name": "Test 4: Skin Problem",
            "symptoms": "skin rash, itching, redness on arms",
            "age": 35, "weight": 68, "sleep": 8, "stress": 3
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'='*50}")
        print(f"📋 {test['name']}")
        print(f"   Symptoms: {test['symptoms']}")
        print(f"   Age: {test['age']}, Sleep: {test['sleep']}h, Stress: {test['stress']}/10")
        print(f"{'='*50}")
        
        # Preprocess
        X_struct, X_text = preprocess_user_input(
            test['symptoms'], test['age'], test['weight'], 
            test['sleep'], test['stress'], 
            tokenizer, scaler, label_encoders, df
        )
        
        if X_struct is None or X_text is None:
            print("❌ Preprocessing failed")
            continue
        
        # Predict
        predictions = model.predict([X_struct, X_text], verbose=0)[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions)[::-1][:5]
        
        print(f"\n🎯 Top Predictions:")
        for i, idx in enumerate(top_indices):
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = predictions[idx] * 100
            bar = "█" * int(confidence / 5) + "░" * (20 - int(confidence / 5))
            print(f"   {i+1}. {disease:<25} {confidence:5.1f}% {bar}")
            
            if i == 0:
                top_disease = disease
                top_confidence = confidence / 100
        
        # Check threshold
        print(f"\n📊 Confidence Check:")
        if top_confidence >= 0.80:
            print(f"   ✅ Confidence {top_confidence*100:.1f}% >= 80% → SHOW DISEASE")
            print(f"   🩺 Predicted Disease: {top_disease}")
        else:
            print(f"   ⚠️ Confidence {top_confidence*100:.1f}% < 80% → HIDE DISEASE")
            print(f"   📝 Showing: 'No Major Health Issue Detected'")
        
        # Get recommendations
        recommendations = get_recommendations(top_disease, df)
        print(f"\n🌿 Ayurvedic Recommendations (based on {top_disease}):")
        print(f"   Herbs: {recommendations.get('herbs', 'N/A')[:80]}")
        print(f"   Diet: {recommendations.get('diet', 'N/A')[:80]}")
        
        results.append({
            "test": test['name'],
            "top_disease": top_disease,
            "confidence": top_confidence,
            "show_disease": top_confidence >= 0.80
        })
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for r in results:
        status = "✅ SHOW" if r['show_disease'] else "⚠️ HIDE"
        print(f"{r['test']:<20} → {r['top_disease']:<25} ({r['confidence']*100:5.1f}%) → {status}")
    
    return results

# ==========================
# MAIN
# ==========================

if __name__ == "__main__":
    print("\n" + "🌿" * 30)
    print("   AYURMEDHA - Model Testing Tool")
    print("🌿" * 30)
    
    # Check if all required files exist
    required_files = [
        MODEL_PATH, TOKENIZER_PATH, SCALER_PATH, 
        LABEL_ENCODER_PATH, LABEL_ENCODERS_PATH, DATA_PATH
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("\n❌ ERROR: Missing required files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease run train_model.py first to generate artifacts.")
        sys.exit(1)
    
    # Load artifacts
    model, tokenizer, scaler, label_encoder, label_encoders, df = load_artifacts()
    
    if model is None:
        print("\n❌ Failed to load model. Cannot run tests.")
        sys.exit(1)
    
    # Run tests
    results = run_tests(model, tokenizer, scaler, label_encoder, label_encoders, df)
    
    print("\n" + "="*60)
    print("✅ Testing Complete!")
    print("="*60)
    print("\n💡 Note: In the actual web app:")
    print("   - Disease is shown ONLY when confidence ≥ 80%")
    print("   - Recommendations are ALWAYS shown (based on top prediction)")
    print("   - This matches the 80% threshold logic in app.py")