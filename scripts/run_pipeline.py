import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess
from src.preprocessing import sentiment_analysis
from src.nlp import ner_extraction
from src.nlp import intent_classifier
from src.nlp import urgency_detector
from src.utils.logger import get_logger

def run_pipeline():
    logger = get_logger("Pipeline")

    print("Step 1: Preprocessing data...")
    cleaned_data = preprocess.run_all()
    logger.info(f"[DEBUG] cleaned_data: {len(cleaned_data)} sources")

    print("Step 2: Sentiment Analysis...")
    sentiment_results = sentiment_analysis.run_all_sentiment()
    logger.info(f"[DEBUG] sentiment_results: {len(sentiment_results)}")

    print("Step 3: Named Entity Recognition...")
    ner_results = ner_extraction.run_ner_all()
    logger.info(f"[DEBUG] ner_results: {len(ner_results) if ner_results else 'None'}")

    print("Step 4: Intent Classification...")
    intent_classifier.run_intent_classification()  # just train
    run_intent_results = intent_classifier.run_intent_classification()  # returns prediction results
    logger.info(f"[DEBUG] intent_run: {len(run_intent_results)}")

    print("Step 5: Urgency Detection...")
    urgency_results = urgency_detector.run()
    logger.info(f"[DEBUG] urgency_results: {len(urgency_results)}")

    print("Step 6: Aggregating results...")
    final_output = []
    index = 0

    for source, df in cleaned_data.items():
        for _, row in df.iterrows():
            final_output.append({
                "source": source,
                "text": row["cleaned_message"],
                "sentiment": sentiment_results[index],
                "entities": ner_results[index],
                "intent": run_intent_results[index],
                "urgency": urgency_results[index]
            })
            index += 1

    print("Pipeline completed. Total records processed:", len(final_output))
    # Optionally save to file
    # import json
    # with open('output/results.json', 'w') as f:
    #     json.dump(final_output, f, indent=4)

if __name__ == "__main__":
    run_pipeline()
