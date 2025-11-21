# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from inference import predict_with_confidence

# BASE PATHS
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except NameError:
    BASE_DIR = Path.cwd().parents[1]

DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------
# UI CONFIG
# ----------------------------------------------------
st.set_page_config(page_title="SmartSpend Categoriser", page_icon="💸")
st.title("💸 SmartSpend – Transaction Categoriser (Hybrid Rules + ML)")

st.write("Enter any raw transaction / merchant string and the system will categorise it automatically.")

merchant = st.text_input("Transaction description (e.g., 'swiggy order 124')")

# ----------------------------------------------------
# PREDICT
# ----------------------------------------------------
if st.button("Predict Category"):
    if merchant.strip() == "":
        st.warning("Please enter a transaction description.")
    else:
        pred_category, conf, explanation = predict_with_confidence(merchant)

        st.success(f"**Prediction:** {pred_category}  \n**Confidence:** {conf:.2f}")

        # ----------- EXPLANATION TABLE -----------
        st.subheader("🧠 Why this prediction?")
        df = pd.DataFrame(explanation, columns=["Token", "Importance"])
        st.table(df)

        # ----------- EXPLANATION BAR CHART -----------
        st.subheader("📊 Feature Importance Chart")

        fig, ax = plt.subplots(figsize=(6, 3))
        tokens = df["Token"]
        scores = df["Importance"]
        ax.barh(tokens, scores)
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Token")
        ax.invert_yaxis()
        st.pyplot(fig)

# ----------------------------------------------------
# EVALUATION DASHBOARD
# ----------------------------------------------------
st.markdown("---")
st.header("📈 Model Evaluation Dashboard")

metrics_path = CONFIG_DIR / "model_metrics.json"

if metrics_path.exists():
    metrics = json.load(open(metrics_path))

    st.subheader("Overall Performance")
    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

    # Per-class F1/Precision/Recall table
    st.subheader("Per-Class Metrics")
    report_df = pd.DataFrame(metrics["report"]).transpose()
    st.dataframe(report_df)

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = np.array(metrics["confusion_matrix"])
    labels = metrics["classes"]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True Label")
    st.pyplot(fig)

else:
    st.info("Model evaluation results will appear here after the next training run.")

# ----------------------------------------------------
# FEEDBACK SECTION
# ----------------------------------------------------
st.markdown("---")
st.subheader("🔁 Help improve SmartSpend")
st.write("If the prediction above was wrong, kindly correct it:")

with open(CONFIG_DIR / "taxonomy.json", "r", encoding="utf-8") as f:
    taxonomy = json.load(f).get("categories", [])

correct_cat = st.selectbox(
    "Correct category (optional):",
    ["(select if wrong prediction)"] + taxonomy
)

# ----------------------------------------------------
# SAVE FEEDBACK
# ----------------------------------------------------
if st.button("Submit Feedback"):
    if merchant.strip() == "" or correct_cat.startswith("("):
        st.warning("Please enter a transaction and select a category.")
    else:
        feedback_path = DATA_DIR / "feedback.csv"

        pred, _, _ = predict_with_confidence(merchant)

        new_row = pd.DataFrame([{
            "merchant": merchant,
            "model_prediction": pred,
            "correct_category": correct_cat
        }])

        if feedback_path.exists():
            new_row.to_csv(feedback_path, mode="a", header=False, index=False)
        else:
            new_row.to_csv(feedback_path, index=False)

        st.success("✅ Feedback saved! Auto-retraining will happen next time the app loads.")
