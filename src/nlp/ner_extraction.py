import os
import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_trf")


# Paths
PROCESSED_PATH = "data/processed"

def extract_entities(text):
    """Extract named entities using spaCy."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def process_file_ner(filename):
    """Apply NER on processed file."""
    file_path = os.path.join(PROCESSED_PATH, filename)
    df = pd.read_csv(file_path)

    df['named_entities'] = df['cleaned_message'].apply(extract_entities)

    output_path = file_path.replace(".csv", "_ner.csv")
    df.to_csv(output_path, index=False)
    print(f"[✅] NER saved to: {output_path}")
    print(df[['cleaned_message', 'named_entities']].head())

def run_ner_all():
    files = [
        "email_cleaned_sentiment.csv",
        "chat_logs_cleaned_sentiment.csv",
        "tickets_cleaned_sentiment.csv"
    ]
    for file in files:
        process_file_ner(file)

if __name__ == "__main__":
    run_ner_all()
