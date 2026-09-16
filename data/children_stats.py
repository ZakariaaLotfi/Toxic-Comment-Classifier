import pandas as pd
import json
import numpy as np

# 1. Load the dataset
df = pd.read_csv("byproduct_data/full_families.csv")

# 2. Deserialize the children column from JSON strings back to Python lists
df["children"] = df["children"].apply(json.loads)

# 3. Create quick-lookup dictionaries
sentiment_score_map = df.set_index("info_order")["practical_sentiment_score"].to_dict()
sentiment_category_map = df.set_index("info_order")["top_sentiment_category"].to_dict()

# 4. Define helper function (Now returning a consistent pd.Series)
def calculate_children_stats(children_list):
    # If the list is empty, return NaN for both columns
    if not children_list:  
        return pd.Series([np.nan, np.nan])
    
    # Retrieve sentiment scores and categories
    scores = [sentiment_score_map.get(child) for child in children_list if child in sentiment_score_map]
    categories = [sentiment_category_map.get(child) for child in children_list if child in sentiment_category_map]
    
    # Filter out NaNs from scores
    valid_scores = [s for s in scores if pd.notna(s)]
    
    # If no valid scores exist, return NaN for both columns
    if not valid_scores:
        return pd.Series([np.nan, np.nan])
    
    # Calculate stats safely
    mean_sentiment = sum(valid_scores) / len(valid_scores)
    
    # Avoid ZeroDivisionError if categories list is empty
    ponos = categories.count("negative") / len(categories) if categories else np.nan
    
    # Return both values inside a Pandas Series
    return pd.Series([mean_sentiment, ponos])

# 5. Apply the function and assign to the two new columns at the same time
df[["children_mean_sentiment", "ponos"]] = df["children"].apply(calculate_children_stats)

# 6. Convert the 'children' column back to a JSON string for CSV storage
df["children"] = df["children"].apply(json.dumps)

# 7. Save the updated DataFrame
df.to_csv("families_with_children_stats.csv", index=False)