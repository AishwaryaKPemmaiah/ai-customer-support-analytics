import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Paths
PROCESSED_DATA_PATH = "data/processed"
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "intent_classifier.pkl")

def load_data():
    df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "all_data_with_labels.csv"))
    return df['message'], df['intent']

def train_model(X, y):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(X, y)
    return pipeline

# ✅ This is the reusable function you can import elsewhere
def train_and_save_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    X, y = load_data()
    model = train_model(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"[✅] Model trained and saved at {MODEL_PATH}")

# ✅ For standalone script execution
if __name__ == "__main__":
    train_and_save_model()
