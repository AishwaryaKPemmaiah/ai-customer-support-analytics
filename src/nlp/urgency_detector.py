# src/nlp/urgency_detector.py

import pandas as pd
import os

PROCESSED_PATH = "data/processed"

HIGH_URGENCY_KEYWORDS = ["asap", "immediately", "urgent", "right away", "now"]
LOW_URGENCY_KEYWORDS = ["just wondering", "not urgent", "whenever", "no rush"]

def detect_urgency(text):
    text = text.lower()
    if any(word in text for word in HIGH_URGENCY_KEYWORDS):
        return "high"
    elif any(word in text for word in LOW_URGENCY_KEYWORDS):
        return "low"
    return "medium"

def process_urgency(filename):
    df = pd.read_csv(os.path.join(PROCESSED_PATH, filename))
    df["urgency"] = df["cleaned_message"].apply(detect_urgency)
    
    out_file = filename.replace(".csv", "_urgency.csv")
    df.to_csv(os.path.join(PROCESSED_PATH, out_file), index=False)
    print(f"[✅] Saved with urgency: {out_file}")
    print(df.head())

if __name__ == "__main__":
    files = ['email_cleaned_sentiment.csv', 'chat_logs_cleaned_sentiment.csv', 'tickets_cleaned_sentiment.csv']
    for file in files:
        process_urgency(file)
