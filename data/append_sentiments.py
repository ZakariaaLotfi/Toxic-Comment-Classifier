import pandas as pd
from transformers import AutoTokenizer, pipeline
import torch

# GPU Check
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU (0)' if device == 0 else 'CPU (-1)'}")

model_path = "models/cardiffnlp-twitter-xlm-roberta-base-sentiment"

# Load Tokenizer explicitly with fixed maximum constraints
print("Loading tokenizer to check token counts...")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    model_max_length=512, 
    truncation=True
)

PATH = "byproduct_data/cleaned.csv"
print(f"Reading {PATH} ...")
df = pd.read_csv(PATH)

# Extract text list from the SAFE dataframe
texts = df["meta_text"].fillna("").astype(str).tolist()

# Initialize the Pipeline passing the pre-configured object
sentiment_task = pipeline(
    "sentiment-analysis",
    model=model_path,
    tokenizer=tokenizer,
    top_k=None,
    device=device,
    batch_size=32
)

print("Starting sentiment analysis pipeline...")
all_predictions = []
chunk_size = 1000

for i in range(0, len(texts), chunk_size):
    chunk = texts[i : i + chunk_size]
    chunk_predictions = sentiment_task(chunk)
    all_predictions.extend(chunk_predictions)
    print(f"Progress: Processed {min(i + chunk_size, len(texts))}/{len(texts)} rows...", flush=True)

print("Pipeline finished. Parsing results...")

positive_scores = []
neutral_scores = []
negative_scores = []
top_categories = []
top_scores = []

for sentiment in all_predictions:
    score_lookup = {entry["label"]: entry["score"] for entry in sentiment}

    positive_scores.append(score_lookup.get("positive", None))
    neutral_scores.append(score_lookup.get("neutral", None))
    negative_scores.append(score_lookup.get("negative", None))

    winning_entry = max(sentiment, key=lambda x: x["score"])
    top_categories.append(winning_entry["label"])
    top_scores.append(winning_entry["score"]) 

sentiments_dict = {
    "top_sentiment_category": top_categories,
    "top_sentiment_score": top_scores,
    "sentiment_positive": positive_scores,
    "sentiment_neutral": neutral_scores,
    "sentiment_negative": negative_scores,
}

sentiments_df = pd.DataFrame(sentiments_dict)

# Concat the sentiment metrics
final_df = pd.concat([df, sentiments_df], axis=1)
final_df.to_csv("threads_with_sentiments.csv", index=False)
print("Job Complete! Saved safely processed data to threads_with_sentiments.csv", flush=True)