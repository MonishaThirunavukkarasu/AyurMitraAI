from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
import pickle
import os
import re
from functools import wraps
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ML preprocessing utilities
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================
# CONFIG
# ==========================

DATA_PATH = "dataset/AyurGenixAI_Dataset.csv"
MAX_WORDS = 10000
MAX_LEN = 50
CONFIDENCE_THRESHOLD = 0.80

# ==========================
# FLASK SETUP
# ==========================

app = Flask(__name__)
app.config["SECRET_KEY"] = "your-secure-random-key-here-change-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///ayurmedha.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."

# ==========================
# DATABASE MODELS
# ==========================

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('PredictionHistory', backref='user', lazy=True)

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    symptoms = db.Column(db.Text, nullable=False)
    age = db.Column(db.Float)
    weight = db.Column(db.Float)
    sleep = db.Column(db.Float)
    stress = db.Column(db.Float)
    predicted_disease = db.Column(db.String(200))
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==========================
# MACHINE LEARNING ARTIFACTS
# ==========================

MODEL_PATH = "model.keras"
TOKENIZER_PATH = "tokenizer.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
LABEL_ENCODERS_PATH = "label_encoders.pkl"

model = None
tokenizer = None
scaler = None
label_encoder = None
label_encoders = None
df = None
symptom_vectorizer = None
symptom_matrix = None
symptom_diseases = None

def normalize_symptoms_text(text):
    """Normalize free-text symptoms for stable tokenization/similarity."""
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

def preprocess_user_input(symptoms, age, weight, sleep, stress):
    """Preprocess user input for model prediction."""
    global tokenizer, scaler, label_encoders, df
    
    if tokenizer is None or scaler is None or label_encoders is None:
        return None, None
    
    if df is None:
        return None, None
    
    all_columns = [col for col in df.columns if col != 'Disease']
    
    user_data = {col: 'unknown' for col in all_columns}
    user_data['Symptoms'] = symptoms
    
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
    df_user['Symptoms_clean'] = df_user['Symptoms'].apply(lambda x: normalize_symptoms_text(x))
    
    symptom_features = df_user['Symptoms'].apply(extract_symptom_features).apply(pd.Series)
    
    cat_cols = [c for c in all_columns if c != 'Symptoms' and c in label_encoders]
    
    encoded = []
    for col in cat_cols:
        if col in label_encoders:
            series = df_user[col].fillna('unknown').astype(str).str.lower()
            le = label_encoders[col]
            enc = np.array([le.transform([v])[0] if v in le.classes_ else 0 for v in series])
            encoded.append(pd.DataFrame({f'{col}_enc': enc}))
    
    structured_df = pd.concat([symptom_features] + encoded, axis=1)
    
    if hasattr(scaler, 'feature_names_in_'):
        expected_columns = scaler.feature_names_in_
        for col in expected_columns:
            if col not in structured_df.columns:
                structured_df[col] = 0
        structured_df = structured_df[expected_columns]
    
    structured_scaled = scaler.transform(structured_df)
    
    seqs = tokenizer.texts_to_sequences(df_user['Symptoms_clean'].tolist())
    text_padded = pad_sequences(seqs, maxlen=MAX_LEN, padding='post')
    
    return structured_scaled.astype(np.float32), text_padded.astype(np.int32)

def find_nearest_disease_by_symptoms(user_symptoms):
    """Find the nearest matching disease from dataset based on symptom similarity"""
    global df, symptom_vectorizer, symptom_matrix, symptom_diseases
    
    if df is None:
        return None, 0
    
    # Build symptom index if not exists
    if symptom_vectorizer is None or symptom_matrix is None:
        subset = df[["Symptoms", "Disease"]].dropna()
        if subset.empty:
            return None, 0
        
        symptoms_series = subset["Symptoms"].astype(str).map(normalize_symptoms_text)
        symptom_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        symptom_matrix = symptom_vectorizer.fit_transform(symptoms_series.tolist())
        symptom_diseases = subset["Disease"].tolist()
    
    # Normalize user symptoms
    query = normalize_symptoms_text(user_symptoms)
    if not query:
        return None, 0
    
    # Calculate similarity
    query_vec = symptom_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, symptom_matrix)[0]
    
    if len(similarities) == 0:
        return None, 0
    
    # Get best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    best_disease = symptom_diseases[best_idx]
    
    return best_disease, best_score

def get_recommendations_from_dataset(disease_name):
    """Get EXACT recommendations from dataset for a disease"""
    global df
    
    if df is None:
        return None
    
    disease_name = str(disease_name).strip().lower()
    
    # Try exact match
    row = df[df["Disease"].str.lower() == disease_name]
    
    # Try partial match
    if row.empty:
        row = df[df["Disease"].str.lower().str.contains(disease_name, na=False)]
    
    if row.empty:
        return None
    
    row = row.iloc[0]
    
    recommendations = {
        "doshas": str(row.get("Doshas", "")).strip() if pd.notna(row.get("Doshas")) else "",
        "herbs": str(row.get("Ayurvedic Herbs", "")).strip() if pd.notna(row.get("Ayurvedic Herbs")) else "",
        "formulation": str(row.get("Formulation", "")).strip() if pd.notna(row.get("Formulation")) else "",
        "diet": str(row.get("Diet and Lifestyle Recommendations", "")).strip() if pd.notna(row.get("Diet and Lifestyle Recommendations")) else "",
        "lifestyle": str(row.get("Yoga & Physical Therapy", "")).strip() if pd.notna(row.get("Yoga & Physical Therapy")) else "",
        "yoga": str(row.get("Yoga & Physical Therapy", "")).strip() if pd.notna(row.get("Yoga & Physical Therapy")) else "",
        "constitution": str(row.get("Constitution/Prakriti", "")).strip() if pd.notna(row.get("Constitution/Prakriti")) else "",
        "prevention": str(row.get("Prevention", "")).strip() if pd.notna(row.get("Prevention")) else "",
        "prognosis": str(row.get("Prognosis", "")).strip() if pd.notna(row.get("Prognosis")) else "",
        "complications": str(row.get("Complications", "")).strip() if pd.notna(row.get("Complications")) else ""
    }
    
    return recommendations if any(recommendations.values()) else None

def load_artifacts():
    """Load all ML artifacts"""
    global model, tokenizer, scaler, label_encoder, label_encoders, df, symptom_vectorizer, symptom_matrix, symptom_diseases
    
    print("=" * 50)
    print("Loading ML Artifacts...")
    print("=" * 50)
    
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH)
            print("✅ Model loaded")
        except Exception as e:
            print(f"❌ Model error: {e}")
    
    if os.path.exists(TOKENIZER_PATH):
        with open(TOKENIZER_PATH, "rb") as f:
            tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded")
    
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print("✅ Scaler loaded")
    
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)
        print(f"✅ Label encoder loaded ({len(label_encoder.classes_)} classes)")
    
    if os.path.exists(LABEL_ENCODERS_PATH):
        with open(LABEL_ENCODERS_PATH, "rb") as f:
            label_encoders = pickle.load(f)
        print(f"✅ Feature encoders loaded ({len(label_encoders)} features)")
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        print(f"✅ Dataset loaded ({len(df)} rows)")
    
    print("=" * 50)

load_artifacts()

# ==========================
# DECORATORS
# ==========================

def prevent_caching(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        resp = view(*args, **kwargs)
        response = make_response(resp)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    return no_cache

# ==========================
# ROUTES
# ==========================

@app.route("/")
@prevent_caching
def index():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        
        if len(password) < 6:
            flash("Password must be at least 6 characters long", "danger")
            return redirect(url_for("signup"))
        
        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("Email already registered. Please login.", "warning")
            return redirect(url_for("login"))
        
        hashed_pw = generate_password_hash(password)
        new_user = User(email=email, password_hash=hashed_pw)
        
        try:
            db.session.add(new_user)
            db.session.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))
        except Exception as e:
            db.session.rollback()
            flash("An error occurred. Please try again.", "danger")
    
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        remember = True if request.form.get("remember") else False
        
        user = User.query.filter_by(email=email).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            flash("Invalid email or password", "danger")
            return redirect(url_for("login"))
        
        login_user(user, remember=remember)
        session.permanent = True
        
        flash(f"Welcome back, {user.email}!", "success")
        next_page = request.args.get("next")
        return redirect(next_page) if next_page else redirect(url_for("dashboard"))
    
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("index"))

@app.route("/dashboard")
@login_required
def dashboard():
    history = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.timestamp.desc())\
        .limit(10)\
        .all() if current_user.is_authenticated else []
    
    return render_template("dashboard.html", history=history)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        try:
            symptoms = request.form.get("symptoms", "").strip()
            age = float(request.form.get("age", 0))
            weight = float(request.form.get("weight", 0))
            sleep = float(request.form.get("sleep", 0))
            stress = float(request.form.get("stress", 0))
            
            if not symptoms:
                flash("Please enter your symptoms", "danger")
                return redirect(url_for("predict"))
            
            # ==========================================
            # FIND NEAREST DISEASE BY SYMPTOM SIMILARITY
            # ==========================================
            nearest_disease, similarity_score = find_nearest_disease_by_symptoms(symptoms)
            
            # Get recommendations for the nearest disease
            recommendations = None
            disease_for_recommendations = None
            
            if nearest_disease:
                recommendations = get_recommendations_from_dataset(nearest_disease)
                disease_for_recommendations = nearest_disease
                print(f"🎯 Nearest disease by symptoms: {nearest_disease} (similarity: {similarity_score:.2%})")
            
            # Also try model prediction if available
            model_predictions = []
            model_top_disease = None
            model_confidence = 0
            
            if model is not None and tokenizer is not None:
                try:
                    X_struct, X_text = preprocess_user_input(symptoms, age, weight, sleep, stress)
                    if X_struct is not None and X_text is not None:
                        preds = model.predict([X_struct, X_text], verbose=0)[0]
                        all_indices = np.argsort(preds)[::-1]
                        
                        for idx in all_indices[:10]:
                            if idx < len(label_encoder.classes_):
                                disease_name = label_encoder.inverse_transform([idx])[0]
                                prob = float(preds[idx])
                                model_predictions.append((disease_name, prob))
                        
                        if model_predictions:
                            model_top_disease = model_predictions[0][0]
                            model_confidence = model_predictions[0][1]
                except Exception as e:
                    print(f"Model prediction error: {e}")
            
            # Decide which disease to use for recommendations
            # Use model if confidence > 30%, otherwise use nearest disease by symptoms
            if model_confidence > 0.3 and model_top_disease:
                final_disease = model_top_disease
                final_confidence = model_confidence
                source = "model"
            elif nearest_disease and similarity_score > 0.1:
                final_disease = nearest_disease
                final_confidence = similarity_score
                source = "symptom_match"
                # Get recommendations if not already got
                if recommendations is None:
                    recommendations = get_recommendations_from_dataset(nearest_disease)
                    disease_for_recommendations = nearest_disease
            else:
                final_disease = "General Wellness"
                final_confidence = 0
                source = "none"
            
            # Ensure we have recommendations
            if recommendations is None and disease_for_recommendations:
                recommendations = get_recommendations_from_dataset(disease_for_recommendations)
            
            # Prepare alternatives for display
            alternatives = []
            if model_predictions:
                for disease, prob in model_predictions[1:6]:
                    alternatives.append({
                        "disease": disease,
                        "confidence": round(prob * 100, 2)
                    })
            elif nearest_disease and similarity_score > 0:
                alternatives.append({
                    "disease": nearest_disease,
                    "confidence": round(similarity_score * 100, 2)
                })
            
            # ==========================================
            # 80% THRESHOLD LOGIC
            # ==========================================
            if final_confidence >= CONFIDENCE_THRESHOLD:
                disease_to_show = final_disease
                confidence_to_show = round(final_confidence * 100, 2)
                show_disease = True
                result_message = f"Based on your symptoms and analysis, the most likely condition is {disease_to_show}."
                print(f"✅ HIGH CONFIDENCE: Showing {disease_to_show}")
            else:
                disease_to_show = "No Major Health Issue Detected"
                confidence_to_show = 0.0
                show_disease = False
                if final_disease != "General Wellness":
                    result_message = f"No specific disease detected above {CONFIDENCE_THRESHOLD*100}% threshold. Showing recommendations for {final_disease} to support your wellness."
                else:
                    result_message = "Based on your symptoms, no specific disease was identified. Showing general wellness recommendations."
                print(f"⚠️ LOW CONFIDENCE: Hiding disease")
            
            # Create explanation
            explanation = {
                "symptom_importance": f"Your symptoms were analyzed and matched most closely with {final_disease if final_disease != 'General Wellness' else 'general wellness patterns'}.",
                "lifestyle_impact": f"Your profile (age: {age}, sleep: {sleep}h, stress: {stress}/10) was considered.",
                "confidence_factors": [
                    f"Top match: {final_disease} ({final_confidence*100:.1f}% confidence)",
                    f"Analysis source: {source}",
                    f"Confidence threshold: {CONFIDENCE_THRESHOLD*100}%"
                ]
            }
            
            # Save to history
            if current_user.is_authenticated:
                history = PredictionHistory(
                    user_id=current_user.id,
                    symptoms=symptoms,
                    age=age,
                    weight=weight,
                    sleep=sleep,
                    stress=stress,
                    predicted_disease=final_disease,
                    confidence=round(final_confidence * 100, 2)
                )
                db.session.add(history)
                db.session.commit()
            
            # Render result with recommendations
            return render_template(
                "result.html",
                disease=disease_to_show,
                confidence=confidence_to_show,
                show_disease=show_disease,
                alternatives=alternatives,
                explanation=explanation,
                result_message=result_message,
                doshas=recommendations.get("doshas", "Information not available in dataset") if recommendations else "Information not available in dataset",
                herbs=recommendations.get("herbs", "Information not available in dataset") if recommendations else "Information not available in dataset",
                formulation=recommendations.get("formulation", "Information not available in dataset") if recommendations else "Information not available in dataset",
                diet=recommendations.get("diet", "Information not available in dataset") if recommendations else "Information not available in dataset",
                lifestyle=recommendations.get("lifestyle", "Information not available in dataset") if recommendations else "Information not available in dataset",
                yoga=recommendations.get("yoga", "Information not available in dataset") if recommendations else "Information not available in dataset",
                constitution=recommendations.get("constitution", "Information not available in dataset") if recommendations else "Information not available in dataset",
                prevention=recommendations.get("prevention", "Information not available in dataset") if recommendations else "Information not available in dataset",
                prognosis=recommendations.get("prognosis", "Information not available in dataset") if recommendations else "Information not available in dataset",
                complications=recommendations.get("complications", "Information not available in dataset") if recommendations else "Information not available in dataset"
            )
            
        except Exception as e:
            flash(f"Prediction error: {str(e)}", "danger")
            print(f"Prediction error: {e}")
            return redirect(url_for("predict"))
    
    return render_template("predict.html")

@app.route("/herbs")
def herbs():
    herb_order = [
        "turmeric", "ashwagandha", "neem", "tulsi", "guduchi", "guggulu", "tripala", "brahmi",
        "amla", "arjuna", "shatavari", "fenugreek", "ginger", "aloe vera", "bhringraj", "gokshura"
    ]

    herb_info = {
        "turmeric": {"desc": "The golden spice with potent anti-inflammatory and antioxidant properties.", "doshas": "Pitta ↓, Kapha ↓", "benefits": ["Joint health", "Skin clarity", "Digestion"], "image": "turmeric.jpg"},
        "ashwagandha": {"desc": "Indian ginseng, a powerful adaptogen for vitality and stress management.", "doshas": "Vata ↓, Pitta ↓", "benefits": ["Energy & stamina", "Nervous system", "Sleep quality"], "image": "ashwagandha.jpg"},
        "neem": {"desc": "A traditional purifier known for skin and immune support.", "doshas": "Pitta ↓, Kapha ↓", "benefits": ["Skin health", "Blood purification", "Immune support"], "image": "neem.jpg"},
        "tulsi": {"desc": "The queen of herbs, revered for adaptogenic and immune-modulating properties.", "doshas": "Vata ↓, Kapha ↓", "benefits": ["Respiratory health", "Stress relief", "Immunity"], "image": "tulsi.jpg"},
        "guduchi": {"desc": "Guduchi supports immunity, detoxification, and resilience.", "doshas": "Pitta ↓, Kapha ↓", "benefits": ["Immunity", "Detox support", "Digestion"], "image": "guduchi.jpg"},
        "guggulu": {"desc": "A resin used in Ayurveda for metabolic and joint support.", "doshas": "Kapha ↓, Vata ↓", "benefits": ["Joint comfort", "Metabolism", "Lipid balance"], "image": "guggulu.jpg"},
        "tripala": {"desc": "A classic three-fruit formulation for digestive cleansing and balance.", "doshas": "Vata ↓, Pitta ↓, Kapha ↓", "benefits": ["Digestion", "Detox", "Regularity"], "image": "tripala.jpg"},
        "brahmi": {"desc": "A brain tonic that supports memory, focus, and calmness.", "doshas": "Vata ↓, Pitta ↓", "benefits": ["Memory", "Mental clarity", "Nervous system"], "image": "brahmi.jpg"},
        "amla": {"desc": "Indian gooseberry rich in antioxidants and rejuvenative benefits.", "doshas": "Pitta ↓, Vata ↓", "benefits": ["Immunity", "Eye health", "Hair health"], "image": "amla.jpg"},
        "arjuna": {"desc": "A valued cardiovascular tonic in classical Ayurveda.", "doshas": "Kapha ↓, Pitta ↓", "benefits": ["Heart support", "Circulation", "Stamina"], "image": "arjuna.jpg"},
        "shatavari": {"desc": "A nourishing herb traditionally used for hormonal and reproductive balance.", "doshas": "Vata ↓, Pitta ↓", "benefits": ["Women's health", "Immunity", "Rejuvenation"], "image": "shatavari.jpg"},
        "fenugreek": {"desc": "A warming seed herb used for digestion and metabolic wellness.", "doshas": "Vata ↓, Kapha ↓", "benefits": ["Digestion", "Metabolism", "Lactation support"], "image": "fenugreek.jpg"},
        "ginger": {"desc": "Universal herb for digestive fire and circulation.", "doshas": "Vata ↓, Kapha ↓", "benefits": ["Digestion", "Nausea relief", "Circulation"], "image": "ginger.jpg"},
        "aloe vera": {"desc": "Cooling botanical known for skin, gut, and pitta balance support.", "doshas": "Pitta ↓", "benefits": ["Skin soothing", "Digestive comfort", "Hydration"], "image": "aloe-vera.jpg"},
        "bhringraj": {"desc": "A rejuvenative herb traditionally used for hair and liver support.", "doshas": "Pitta ↓, Vata ↓", "benefits": ["Hair wellness", "Liver support", "Calming"], "image": "bhringraj.jpg"},
        "gokshura": {"desc": "A strengthening herb used for urinary and vitality support.", "doshas": "Vata ↓", "benefits": ["Urinary support", "Vitality", "Strength"], "image": "gokshura.jpg"}
    }

    herbs_data = []
    for herb in herb_order:
        info = herb_info.get(herb, {})
        herbs_data.append({
            "name": herb.title(),
            "description": info.get("desc", ""),
            "doshas": info.get("doshas", ""),
            "benefits": info.get("benefits", []),
            "image": info.get("image", "default-herb.jpg")
        })

    return render_template("herbs.html", herbs=herbs_data)

@app.route("/dosha")
def dosha():
    return render_template("dosha.html")

@app.route("/api/prediction-history")
@login_required
def prediction_history():
    history = PredictionHistory.query.filter_by(user_id=current_user.id)\
        .order_by(PredictionHistory.timestamp.desc())\
        .limit(20)\
        .all()
    
    data = [{
        "id": h.id,
        "disease": h.predicted_disease,
        "confidence": h.confidence,
        "timestamp": h.timestamp.strftime("%Y-%m-%d %H:%M")
    } for h in history]
    
    return jsonify(data)

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model is not None,
        "threshold": CONFIDENCE_THRESHOLD
    })

# ==========================
# ERROR HANDLERS
# ==========================

@app.errorhandler(404)
def not_found_error(error):
    return render_template("404.html"), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template("500.html"), 500

# ==========================
# CREATE TABLES
# ==========================

with app.app_context():
    db.create_all()
    print("✅ Database tables created/verified")

if __name__ == "__main__":
    print("=" * 60)
    print("🌿 AYURMEDHA - Ayurvedic Intelligence Platform")
    print("=" * 60)
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD * 100}%")
    print(f"📊 Model Status: {'✅ Loaded' if model else '❌ Not Loaded'}")
    print(f"🌐 Starting server at: http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)