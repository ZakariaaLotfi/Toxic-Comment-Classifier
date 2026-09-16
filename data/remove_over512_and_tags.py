import pandas as pd
from transformers import AutoTokenizer
import re

PATH = "byproduct_data/sorted.csv"
print(f"Reading {PATH} ...")
df = pd.read_csv(PATH)

# Define the tags we want to pull out
exclusion_tags = ["exclude_empty", "exclude_lang", "exclude_bot", "exclude_image"]

print("Isolating and saving rows with exclusion tags ...")
# 1. Capture the rows matching your exclusion tags and save them immediately
df_excluded = df[df["split"].isin(exclusion_tags)].copy()
if len(df_excluded) > 0:
    df_excluded.to_csv("excluded_tags.csv", index=False)
    print(f"Saved {len(df_excluded)} rows matching exclusion tags to 'excluded_tags.csv'")
else:
    print("No rows matched the exclusion tags.")

# 2. Filter the main dataframe to keep ONLY the safe/included rows
df = df[~df["split"].isin(exclusion_tags)].reset_index(drop=True)

def clean_social_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase everything to make matching easier
    text = text.lower()
    
    # 2. Remove common Reddit/platform moderation placeholders
    text = text.replace("[removed]", "").replace("[deleted]", "").replace("[linebreak]", "")
    
    # 3. Remove URLs (http, https, and www.)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 4. Clean up extra whitespace left behind
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

print("Cleaning links and reddit placeholders ...")
df["meta_text"] = df["meta_text"].fillna("").apply(clean_social_text)

# Load Tokenizer explicitly with fixed maximum constraints
print("Loading tokenizer to check token counts...")

model_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(
    model_path, 
    model_max_length=512, 
    truncation=True
)

# Helper function to count tokens safely
def get_token_length(text):
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text, add_special_tokens=True))

print("Calculating token lengths for all rows...")
token_lengths = df["meta_text"].fillna("").apply(get_token_length)

# 3. Split the DataFrame based on the 512 token limit
df_long = df[token_lengths > 512].copy()
df_safe = df[token_lengths <= 512].reset_index(drop=True)

print(f"Analysis complete: Found {len(df_long)} rows exceeding 512 tokens.")

# Save the long rows immediately to their own verification file
if len(df_long) > 0:
    df_long.to_csv("long_sequences.csv", index=False)
    print("Saved rows exceeding 512 tokens to 'long_sequences.csv'")

print("Saved final cleaned data to cleaned.csv")
df_safe.to_csv("cleaned.csv", index=False)