
# ⭐ **SMARTSPEND — Automated AI-Based Financial Transaction Categorisation**

SmartSpend is an AI-powered system that automatically categorises noisy and unstructured financial transaction descriptions into meaningful categories like **Food**, **Groceries**, **Fuel**, **Shopping**, **Travel**, **Bills & Utilities**, etc.

This project was developed as part of the **GHCI AI Hackathon**.

----------

## 📌 **What SmartSpend Does**

SmartSpend cleans raw merchant descriptions and predicts the correct spending category using:

-   **Rule Engine** → Detects known merchants
    
-   **Machine Learning Model (TF-IDF + SVM)** → For unseen/noisy text
    
-   **Hybrid Decision Logic** → Combines both outputs
    
-   **Explainability** → Shows top keywords influencing the prediction
    
-   **Auto-Retraining** → Learns from user feedback in Streamlit
    

----------

## 📁 **Folder Structure**
```
SmartSpend/
│
├── app/
│   ├── preprocess.py          # Text cleaning + normalization
│   ├── inference.py           # Rule engine + ML prediction logic
│   ├── train_hybrid.py        # Training + auto-retraining
│   └── streamlit_app.py       # Streamlit UI
│
├── data/
│   ├── transactions.csv       # Base labelled dataset
│   ├── feedback.csv           # User corrections from UI
│   └── feedback_version.txt   # Tracks retraining state
│
├── config/
│   └── taxonomy.json          # List of supported categories
|   └── model matix            # store the parameters of ML model 
│
└── saved_model/
|   ├── svm_model.pkl          # Trained machine learning model
|   └── tfidf_vectorizer.pkl   # TF-IDF vectorizer
| 
└── SMARTSPEND REPORT
 ```
----------

## ▶️ **How to Run the Project**

Since this project does not use a `requirements.txt`, install common dependencies manually:

````
pip install streamlit
pip install scikit-learn
pip install pandas
pip install numpy
```` 

Then launch the app:

`streamlit run app/streamlit_app.py` 

----------

## 🧪 **How to Retrain the Model**

If you want to manually retrain:

`python app/train_hybrid.py` 

The system will:

-   Load transactions + feedback
    
-   Retrain TF-IDF + SVM
    
-   Update taxonomy
    
-   Save updated model files
    

----------
##  **TEAM MEMBERS**

        Tharun babu C
        Amirtha K
        
## 🙏 **Thank You for Reviewing Our Project**

We appreciate your time!  
SmartSpend is built to demonstrate real-world AI + ML application in fintech.

----------
