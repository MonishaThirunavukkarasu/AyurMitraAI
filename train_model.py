"""
train_model.py
Fixed version - resolves the sample mismatch between structured and text data
"""

import os
import pickle
import re

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ==========================
# CONFIGURATION
# ==========================

DATA_PATH = "dataset/AyurGenixAI_Dataset.csv"
MAX_WORDS = 10000
MAX_LEN = 50
EMBEDDING_DIM = 128
EPOCHS = 50
BATCH_SIZE = 32
TEST_SIZE = 0.2
RANDOM_STATE = 42

TOKENIZER_PATH = "tokenizer.pkl"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.pkl"
MODEL_PATH = "model.keras"
LABEL_ENCODERS_PATH = "label_encoders.pkl"

TEXT_COLUMN = "Symptoms"
TARGET_COLUMN = "Disease"
COMPLETE_FEATURES = [
    'Symptoms', 'Age Group', 'Gender', 'Occupation and Lifestyle',
    'Family History', 'Symptom Severity', 'Medical History', 'Current Medications',
    'Risk Factors', 'Environmental Factors', 'Sleep Patterns', 'Stress Levels',
    'Physical Activity Levels', 'Constitution/Prakriti', 'Seasonal Variation',
    'Dietary Habits', 'Allergies (Food/Env)'
]

# ==========================
# DATA PREPROCESSOR CLASS
# ==========================

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
        self.max_len = MAX_LEN

    def clean_text(self, text):
        if pd.isna(text):
            return "unknown"
        text = str(text).lower().strip()
        text = re.sub(r'[^a-zA-Z0-9\s,]', ' ', text)
        return ' '.join(text.split())

    def extract_features(self, text):
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

    def preprocess(self, df, fit=True):
        # Clean symptoms
        df = df.copy()
        df['Symptoms_clean'] = df['Symptoms'].apply(self.clean_text)
        
        # Extract numeric features from symptoms
        symptom_features = df['Symptoms'].apply(self.extract_features).apply(pd.Series)
        
        # Tokenize for LSTM
        if fit:
            self.tokenizer.fit_on_texts(df['Symptoms_clean'])
        seqs = self.tokenizer.texts_to_sequences(df['Symptoms_clean'])
        symptom_padded = pad_sequences(seqs, maxlen=self.max_len, padding='post')
        
        # Encode categorical features
        cat_cols = [c for c in df.columns if c not in ['Symptoms', 'Symptoms_clean', 'Disease']]
        encoded = []
        
        for col in cat_cols:
            if col in df.columns:
                series = df[col].fillna('unknown').astype(str).str.lower()
                if col in self.label_encoders and not fit:
                    le = self.label_encoders[col]
                    enc = np.array([le.transform([v])[0] if v in le.classes_ else 0 for v in series])
                else:
                    le = LabelEncoder()
                    enc = le.fit_transform(series)
                    if fit:
                        self.label_encoders[col] = le
                encoded.append(pd.DataFrame({f'{col}_enc': enc}))
        
        # Combine all structured features
        structured_df = pd.concat([symptom_features] + encoded, axis=1)
        
        # Scale
        if fit:
            structured_scaled = self.scaler.fit_transform(structured_df)
        else:
            structured_scaled = self.scaler.transform(structured_df)
        
        return {
            'structured': structured_scaled.astype(np.float32),
            'text': symptom_padded.astype(np.int32),
            'feature_names': structured_df.columns.tolist()
        }

# ==========================
# 1. LOAD DATA
# ==========================

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset contains {len(df)} rows")

# Use available features
available_features = [f for f in COMPLETE_FEATURES if f in df.columns]
df = df[available_features + [TARGET_COLUMN]].copy()
print(f"✓ Using {len(available_features)} features")

# Handle rare classes
class_dist = df[TARGET_COLUMN].value_counts()
keep_top_n = 8
original_top_classes = class_dist.nlargest(keep_top_n).index.tolist()

rare_classes = class_dist[class_dist < 5].index.difference(original_top_classes).tolist()
if rare_classes:
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 'Other Diseases' if x in rare_classes else x)
    print(f"✓ Combined {len(rare_classes)} rare classes into 'Other Diseases'")

# Keep only rows with non-null symptoms (CRITICAL FIX)
df = df.dropna(subset=['Symptoms'])
print(f"✓ After dropping rows with missing symptoms: {len(df)} rows")

class_dist = df[TARGET_COLUMN].value_counts()
allowed_classes = [c for c in original_top_classes if c in class_dist.index]
if 'Other Diseases' in class_dist.index and 'Other Diseases' not in allowed_classes:
    allowed_classes.append('Other Diseases')

if len(allowed_classes) == 0:
    allowed_classes = class_dist.index.tolist()

print(f"Using allowed classes: {allowed_classes}")
df = df[df[TARGET_COLUMN].isin(allowed_classes)].copy()

# Limit dominance of Other Diseases
if 'Other Diseases' in df[TARGET_COLUMN].values:
    other_df = df[df[TARGET_COLUMN] == 'Other Diseases']
    non_other_df = df[df[TARGET_COLUMN] != 'Other Diseases']
    max_other = int(max(1, 0.25 * len(df)))
    if len(other_df) > max_other:
        other_df = other_df.sample(max_other, random_state=RANDOM_STATE)
        df = pd.concat([non_other_df, other_df], ignore_index=True)
        print(f"✓ Undersampled Other Diseases to {max_other} records")

print(f"Final class distribution:\n{df[TARGET_COLUMN].value_counts()}")

# ==========================
# 2. PREPROCESSING
# ==========================

preprocessor = DataPreprocessor()
processed = preprocessor.preprocess(df, fit=True)

X_struct = processed['structured']
X_text = processed['text']

# CRITICAL FIX: Ensure X_struct and X_text have same number of samples
print(f"\nBefore alignment - X_struct: {X_struct.shape}, X_text: {X_text.shape}")

# Take the minimum number of samples
min_samples = min(X_struct.shape[0], X_text.shape[0])
X_struct = X_struct[:min_samples]
X_text = X_text[:min_samples]
df = df.iloc[:min_samples].copy()

print(f"After alignment - X_struct: {X_struct.shape}, X_text: {X_text.shape}")

# Label encoding for target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET_COLUMN])
n_classes = len(np.unique(y))

print(f"✓ Preprocessed data: X_struct {X_struct.shape}, X_text {X_text.shape}, Classes: {n_classes}")

# Persist artifacts
with open(TOKENIZER_PATH, "wb") as f:
    pickle.dump(preprocessor.tokenizer, f)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(preprocessor.scaler, f)
with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)
with open(LABEL_ENCODERS_PATH, "wb") as f:
    pickle.dump(preprocessor.label_encoders, f)

print(f"✓ Label encoder saved with classes: {list(label_encoder.classes_)}")

# ==========================
# 3. SKIP SMOTE FOR SMALL DATASET (to avoid errors)
# ==========================

print("\nUsing original data (skipping SMOTE due to small dataset)...")
X_struct_bal = X_struct
X_text_bal = X_text
y_bal = y

print(f"Total samples: {len(y_bal)}")

# ==========================
# 4. Train/Test Split (Fixed for small dataset)
# ==========================

# For small datasets, use a smaller test size or fallback to all data
n_samples_total = len(X_struct_bal)
n_classes_total = len(np.unique(y_bal))

# Calculate minimum test size needed for stratification
min_test_needed = min(n_classes_total, 3)  # At least 3 or number of classes

if n_samples_total - min_test_needed < 2:
    # Not enough data - use all for training
    print(f"⚠️ Not enough data for test split. Using all {n_samples_total} samples for training.")
    X_struct_train, X_struct_test = X_struct_bal, X_struct_bal
    X_text_train, X_text_test = X_text_bal, X_text_bal
    y_train, y_test = y_bal, y_bal
else:
    test_size = min(0.2, min_test_needed / n_samples_total)
    try:
        X_struct_train, X_struct_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_struct_bal, X_text_bal, y_bal, test_size=test_size, random_state=RANDOM_STATE, stratify=y_bal
        )
        print(f"Training on {len(X_struct_train)} samples, testing on {len(X_struct_test)} samples")
    except ValueError as e:
        print(f"⚠️ Stratified split failed: {e}")
        print(f"Using non-stratified split...")
        X_struct_train, X_struct_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
            X_struct_bal, X_text_bal, y_bal, test_size=0.2, random_state=RANDOM_STATE
        )
        print(f"Training on {len(X_struct_train)} samples, testing on {len(X_struct_test)} samples")

# ==========================
# 5. MODEL ARCHITECTURE
# ==========================

# Structured branch (ANN)
struct_in = Input(shape=(X_struct_train.shape[1],), name='structured')
s = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(struct_in)
s = BatchNormalization()(s)
s = Dropout(0.3)(s)
s = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(s)
s = BatchNormalization()(s)
s = Dropout(0.3)(s)
s = Dense(64, activation='relu')(s)

# Text branch (LSTM)
text_in = Input(shape=(MAX_LEN,), name='text')
t = Embedding(MAX_WORDS, EMBEDDING_DIM, mask_zero=True)(text_in)
t = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(t)
t = Dense(64, activation='relu')(t)
t = Dropout(0.3)(t)

# Combine
combined = Concatenate()([s, t])
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(combined)
x = BatchNormalization()(x)
x = Dropout(0.4)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=[struct_in, text_in], outputs=output)
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================
# 6. TRAINING
# ==========================

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

print("Starting training...")

# Only use validation if test set is different from training set
if len(X_struct_test) > 0 and not np.array_equal(X_struct_train, X_struct_test):
    history = model.fit(
        [X_struct_train, X_text_train], y_train,
        validation_data=([X_struct_test, X_text_test], y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
else:
    history = model.fit(
        [X_struct_train, X_text_train], y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

# ==========================
# 7. EVALUATION
# ==========================

print("Evaluating model...")
if len(X_struct_test) > 0 and not np.array_equal(X_struct_train, X_struct_test):
    y_pred = model.predict([X_struct_test, X_text_test])
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    class_names = label_encoder.inverse_transform(np.unique(np.concatenate([y_test, y_pred_classes])))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=class_names, zero_division=0))
else:
    print("⚠️ No separate test set available for evaluation")

# ==========================
# 8. SAVE MODEL
# ==========================
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

print("All training steps completed.")