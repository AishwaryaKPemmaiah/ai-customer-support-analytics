import pandas as pd
import os

training_samples = [
    {"message": "I can't log in", "intent": "login_issue"},
    {"message": "My password reset link isn't working", "intent": "login_issue"},
    {"message": "Please fix this billing error", "intent": "billing"},
    {"message": "Need clarification on my invoice", "intent": "billing"},
    {"message": "Your service is amazing!", "intent": "feedback"},
    {"message": "I love your support", "intent": "feedback"},
    {"message": "Why was my card charged twice?", "intent": "billing"},
    {"message": "Still no update on my issue", "intent": "complaint"},
    {"message": "Thanks for your help", "intent": "feedback"},
]

df = pd.DataFrame(training_samples)

os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/all_data_with_labels.csv", index=False)
print("[✅] Training data created: data/processed/all_data_with_labels.csv")
