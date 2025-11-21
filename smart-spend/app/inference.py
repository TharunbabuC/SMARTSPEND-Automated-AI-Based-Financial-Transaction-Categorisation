# app/inference.py

import json
import joblib
import numpy as np
from pathlib import Path
import pandas as pd

from preprocess import clean_text, keyword_boost
from train_hybrid import train_model   # auto-retrain engine

# -------------------------------
# Paths
# -------------------------------
try:
    FILE_DIR = Path(__file__).resolve().parent
except NameError:
    FILE_DIR = Path.cwd()

BASE_DIR = FILE_DIR.parents[0]
MODEL_DIR = BASE_DIR / "saved_model"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"

FEEDBACK_FILE = DATA_DIR / "feedback.csv"
VERSION_FILE = DATA_DIR / "feedback_version.txt"

# -------------------------------------------------
# AUTO RETRAIN CHECK
# -------------------------------------------------
def retrain_if_needed():
    """Retrain ONLY if feedback.csv changed."""
    if not FEEDBACK_FILE.exists():
        return

    fb_rows = max(0, sum(1 for _ in open(FEEDBACK_FILE, "r", encoding="utf-8")) - 1)

    if VERSION_FILE.exists():
        prev_rows = int(open(VERSION_FILE).read().strip())
    else:
        prev_rows = 0

    if fb_rows == prev_rows:
        return

    print("🔄 New feedback detected → Retraining model...")
    train_model()

    with open(VERSION_FILE, "w") as f:
        f.write(str(fb_rows))

    print("✅ Retraining complete.")


# RUN AUTO-RETRAIN
retrain_if_needed()

# -------------------------------------------------
# LOAD MODEL & VECTORIZER
# -------------------------------------------------
model = joblib.load(MODEL_DIR / "svm_model.pkl")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")

with open(CONFIG_DIR / "taxonomy.json", "r", encoding="utf-8") as f:
    TAXONOMY = json.load(f).get("categories", list(model.classes_))


# -------------------------------------------------
# FEATURE NAME EXTRACTION (FeatureUnion)
# -------------------------------------------------
def get_feature_names(v):
    names = []
    for name, vec in v.transformer_list:
        try:
            fn = vec.get_feature_names_out()
            fn = [f"{name}::{x}" for x in fn]
            names.extend(fn)
        except:
            pass
    return np.array(names)


# -------------------------------------------------
# EXTRACT TRUE LinearSVC FROM CalibratedClassifierCV
# -------------------------------------------------
def get_linear_svc(model):
    if hasattr(model, "calibrated_classifiers_"):
        return model.calibrated_classifiers_[0].estimator
    return None


# -------------------------------------------------
# EXPLANATION ENGINE
# -------------------------------------------------
def explain_prediction(text):
    vec = vectorizer.transform([text]).toarray()[0]
    feature_names = get_feature_names(vectorizer)

    svc = get_linear_svc(model)
    if svc is None or not hasattr(svc, "coef_"):
        return [("no-explanation", 0.0)]

    coef = svc.coef_[0]

    active = np.where(vec > 0)[0]
    if len(active) == 0:
        return [("no-explanation", 0.0)]

    importance = []
    for idx in active:
        score = coef[idx] * vec[idx]
        importance.append((feature_names[idx], float(score)))

    importance = sorted(importance, key=lambda x: abs(x[1]), reverse=True)

    return importance[:10]


# -------------------------------------------------
# MAIN PREDICTION FUNCTION
# -------------------------------------------------
def predict_with_confidence(raw_text: str):
    cleaned = clean_text(raw_text)

    # 1. RULE BOOST
    r = keyword_boost(cleaned)
    if r:
        return r, 0.95, [("rule-match", 1.0)]

    # 2. ML PREDICTION
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]

    best_idx = np.argmax(probs)
    category = model.classes_[best_idx]
    conf = float(probs[best_idx])

    # 3. CONFIDENCE STABILIZER
    if conf < 0.55:
        conf += 0.20
    conf = min(conf, 0.97)

    # 4. EXPLANATION
    explanation = explain_prediction(cleaned)

    return category, conf, explanation
