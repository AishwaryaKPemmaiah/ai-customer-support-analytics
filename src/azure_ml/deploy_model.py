import os
import joblib

# ✅ Match the model path used in train_model.py
MODEL_PATH = os.path.join("data", "models", "intent_classifier.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    
    model = joblib.load(MODEL_PATH)
    print("[✅] Model loaded successfully.")
    return model

if __name__ == "__main__":
    model = load_model()

    # ✅ Example prediction
    sample_text = ["I need help resetting my password"]
    prediction = model.predict(sample_text)
    
    print(f"[📌] Prediction for sample text: '{sample_text[0]}' → {prediction[0]}")
