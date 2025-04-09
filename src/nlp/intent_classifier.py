# src/nlp/intent_classifier.py

import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

PROCESSED_PATH = "data/processed"
MODEL_PATH = "data/models"

def train_intent_classifier():
    # Example labeled data (in real cases, you'll use a labeled dataset)
    data = [
        ("I can't log in", "login_issue"),
        ("Need to reset password", "password_reset"),
        ("Where's my order?", "order_tracking"),
        ("Billing error in invoice", "billing_issue"),
        ("How to cancel order?", "cancel_order")
    ]
    df = pd.DataFrame(data, columns=["text", "intent"])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["text"])
    y = df["intent"]

    clf = LogisticRegression()
    clf.fit(X, y)

    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(clf, f"{MODEL_PATH}/intent_model.pkl")
    joblib.dump(vectorizer, f"{MODEL_PATH}/intent_vectorizer.pkl")
    print("[✅] Intent classifier trained and saved.")

if __name__ == "__main__":
    train_intent_classifier()
