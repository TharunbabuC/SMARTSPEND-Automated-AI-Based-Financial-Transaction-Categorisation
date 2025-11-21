# app/train_hybrid.py

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_model"
CONFIG_DIR = BASE_DIR / "config"

DATASET = DATA_DIR / "transactions.csv"
FEEDBACK = DATA_DIR / "feedback.csv"

def train_model():
    print("⏳ Training model...")

    # -----------------------------
    # Load main dataset
    # -----------------------------
    df = pd.read_csv(DATASET)

    # Add feedback data if exists
    if FEEDBACK.exists():
        fb = pd.read_csv(FEEDBACK)
        fb.columns = ["merchant", "model_prediction", "correct_category"]
        fb = fb.rename(columns={"correct_category": "category"})
        df = pd.concat([df, fb[["merchant", "category"]]], ignore_index=True)

    df = df.dropna()

    X = df["merchant"].astype(str)
    y = df["category"].astype(str)

    # -----------------------------
    # Vectorizer (word + char)
    # -----------------------------
    word_tfidf = ("word", TfidfVectorizer(
        analyzer="word",
        ngram_range=(1,2),
        min_df=3,
        stop_words="english"
    ))

    char_tfidf = ("char", TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3,5),
        min_df=3
    ))

    vectorizer = FeatureUnion([word_tfidf, char_tfidf])
    X_vec = vectorizer.fit_transform(X)

    # -----------------------------
    # Model = Linear SVM + Prob Calibrator
    # -----------------------------
    base = LinearSVC(max_iter=20000)
    model = CalibratedClassifierCV(base, method="sigmoid", cv=4)

    X_train, X_val, y_train, y_val = train_test_split(
        X_vec, y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    # -----------------------------
    # Print metrics in terminal
    # -----------------------------
    val_acc = accuracy_score(y_val, pred)
    print("\n📊 Validation accuracy:", val_acc)
    print(classification_report(y_val, pred))

    # -----------------------------
    # SAVE MODEL + VECTORIZER
    # -----------------------------
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "svm_model.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")

    # -----------------------------
    # Save taxonomy
    # -----------------------------
    taxonomy = {"categories": sorted(list(model.classes_))}
    with open(CONFIG_DIR / "taxonomy.json", "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)

    # -----------------------------
    # SAVE METRICS FOR STREAMLIT DASHBOARD
    # -----------------------------
    cm = confusion_matrix(y_val, pred)
    report = classification_report(y_val, pred, output_dict=True)

    metrics = {
        "accuracy": float(val_acc),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "classes": taxonomy["categories"]
    }

    with open(CONFIG_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("📁 Saved evaluation to config/model_metrics.json")
    print("✔ Training completed.\n")

if __name__ == "__main__":
    train_model()
