import numpy as np
import pandas as pd

# Load the dataset with proper string path
df = pd.read_csv("byproduct_data/threads_with_sentiments.csv")

# Define column names and condition strings as proper variables
tsc = "top_sentiment_score"
tsc_label = "top_sentiment_category"

# Calculate practical sentiment scores using np.select (cleaner than nested np.where)
conditions = [
    df[tsc_label] == "neutral",
    df[tsc_label] == "negative",
    df[tsc_label] == "positive"
]
choices = [0.0, -df[tsc], df[tsc]]

practical_scores = np.select(conditions, choices, default=df[tsc])

# Fix the bug: Assign the array to a single, specific column name string
df["practical_sentiment_score"] = practical_scores

# Export the updated DataFrame with proper string path
df.to_csv("threads_with_sentiments.csv", index=False)
