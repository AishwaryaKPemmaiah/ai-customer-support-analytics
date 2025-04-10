import os
import pandas as pd
from src.azure_ml.train_model import train_and_save_model

from src.azure_ml.predict_intent import predict_intent  # this should already use the trained model
from src.utils.logger import get_logger

logger = get_logger("IntentClassifier")

PROCESSED_PATH = "data/processed"

def train_intent_classifier():
    # Train and save model
    train_and_save_model()

    results = []

    files = [
        "email_cleaned_sentiment.csv",
        "chat_logs_cleaned_sentiment.csv",
        "tickets_cleaned_sentiment.csv"
    ]

    for file in files:
        input_path = os.path.join(PROCESSED_PATH, file)
        df = pd.read_csv(input_path)

        # Predict intents for each cleaned message
        intents = [predict_intent(text) for text in df["cleaned_message"]]
        df["intent"] = intents
        results.extend(intents)  # collect intents in flat list

        # Save new file with intent
        output_path = input_path.replace(".csv", "_intent.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"[✅] Saved with intent: {os.path.basename(output_path)}")
        logger.info(df.head(2).to_string(index=False))

    return results
