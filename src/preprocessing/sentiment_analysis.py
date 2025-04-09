# src/preprocessing/sentiment_analysis.py

import os
import pandas as pd
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

sid = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = sid.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def process_file_sentiment(filename):
    input_path = os.path.join(PROCESSED_DATA_PATH, filename)
    df = pd.read_csv(input_path)
    
    df['sentiment'] = df['cleaned_message'].apply(analyze_sentiment)

    output_path = os.path.join(PROCESSED_DATA_PATH, filename.replace('.csv', '_sentiment.csv'))
    df.to_csv(output_path, index=False)
    print(f"[✅] Saved with sentiment: {output_path}")
    print(df.head())  # Show first 5 rows after sentiment


def run_all_sentiment():
    files = ['email_cleaned.csv', 'chat_logs_cleaned.csv', 'tickets_cleaned.csv']
    for file in files:
        process_file_sentiment(file)

if __name__ == "__main__":
    run_all_sentiment()
