'''# src/preprocessing/preprocess.py

import os
import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def preprocess_file(filename):
    full_path = os.path.join(RAW_DATA_PATH, filename)
    with open(full_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['cleaned_message'] = df['message'].apply(clean_text)

    output_file = os.path.join(PROCESSED_DATA_PATH, filename.replace('.json', '_cleaned.csv'))
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

def run_all():
    files = ['emails.json', 'chat_logs.json', 'tickets.json']
    for file in files:
        preprocess_file(file)

if __name__ == "__main__":
    run_all()
'''

"""
import os
import json
import pandas as pd
import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def clean_text(text):
"""
    #Clean and preprocess text using spaCy.
    #- Lowercase
    #- Remove punctuation, stopwords, and non-alphabetic tokens
"""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)

def preprocess_file(filename):
    full_path = os.path.join(RAW_DATA_PATH, filename)
    with open(full_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['cleaned_message'] = df['message'].apply(clean_text)

    output_file = os.path.join(PROCESSED_DATA_PATH, filename.replace('.json', '_cleaned.csv'))
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved preprocessed data to: {output_file}")

def run_all():
    files = ['email.json', 'chat_logs.json', 'tickets.json']
    for file in files:
        preprocess_file(file)

if __name__ == "__main__":
    run_all()
"""



# src/preprocessing/preprocess.py

import os
import json
import pandas as pd
import spacy
import re

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"

def clean_text(text):
    """
    Clean and preprocess text using spaCy.
    - Lowercase
    - Remove punctuation, stopwords, and non-alphabetic tokens
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text.strip())  # Remove extra spaces
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and token.is_alpha
    ]
    return " ".join(tokens)

def preprocess_file(filename):
    """
    Loads a raw JSON file, cleans the messages, saves the output CSV,
    and returns the cleaned DataFrame.
    """
    full_path = os.path.join(RAW_DATA_PATH, filename)
    with open(full_path, 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    df['cleaned_message'] = df['message'].apply(clean_text)

    output_file = os.path.join(PROCESSED_DATA_PATH, filename.replace('.json', '_cleaned.csv'))
    df.to_csv(output_file, index=False)
    print(f"[INFO] Saved preprocessed data to: {output_file}")
    
    return df

def run_all():
    """
    Runs preprocessing on all supported files and returns a dictionary of cleaned DataFrames.
    """
    files = {
        'email': 'email.json',
        'chat': 'chat_logs.json',
        'tickets': 'tickets.json'
    }

    cleaned_data = {}

    for source, file in files.items():
        df = preprocess_file(file)
        cleaned_data[source] = df

    return cleaned_data

if __name__ == "__main__":
    run_all()
