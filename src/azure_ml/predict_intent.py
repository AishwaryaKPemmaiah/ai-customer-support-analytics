import joblib
import os

MODEL_PATH = "data/models/intent_classifier.pkl"

# Load the model once (global scope)
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise Exception(f"❌ Model not found at {MODEL_PATH}. Please train the model first.")

def predict_intent(text):
    """
    Predict the intent for a single text message.
    """
    return model.predict([text])[0]
