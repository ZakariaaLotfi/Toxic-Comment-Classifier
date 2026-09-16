import pandas as pd
import csv
import re

df = pd.read_csv("byproduct_data/sorted.csv")

df = df[~df["split"].isin(["exclude_empty","exclude_lang","exclude_bot","exclude_image"])]

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

# 2. Define the contextual thread sorting logic
def reddit_thread_key(series):
    def parse_value(s):
        # Handle NaN/missing values safely
        if pd.isna(s) or not isinstance(s, str):
            return (-1, 0, []) 
            
        s = s.strip()
        
        # 1. Handle Titles (Priority 0)
        if '-title' in s:
            try:
                parent_id = int(s.replace('-title', ''))
                return (parent_id, 0, [])
            except ValueError:
                return (-1, 0, [s])
                
        # 2. Handle Posts (Priority 1)
        elif '-post' in s:
            try:
                parent_id = int(s.replace('-post', ''))
                return (parent_id, 1, [])
            except ValueError:
                return (-1, 1, [s])
                
        # 3. Handle Reply Threads (Priority 2)
        else:
            try:
                nums = list(map(int, s.split(',')))
                return (nums[0], 2, nums[1:])
            except ValueError:
                # Catch-all fallback for any other unstructured strings
                return (-1, 3, [s]) 
            
    return series.apply(parse_value)

print("Cleaning text column...")
df["meta_text"] = df["meta_text"].fillna("").apply(clean_social_text)

# 3. Sort the DataFrame
sorted_df = df.sort_values(by='info_order', key=reddit_thread_key)

sorted_df.to_csv("cleaned.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)