import ast
import numpy as np
import pandas as pd

df = pd.read_csv("byproduct_data/families_with_children_stats.csv")
df["meta_date"] = pd.to_datetime(df["meta_date"])

# 1. Map ID directly to its datetime in a dictionary for O(1) instant lookups
id_to_date = dict(zip(df["info_order"], df["meta_date"]))


# 2. Define a clean, standalone function for row processing
def calculate_row_delays(row):
    raw_children = row["children"]

    # Safely parse
    if isinstance(raw_children, str):
        try:
            child_ids = ast.literal_eval(raw_children)
        except (ValueError, SyntaxError):
            return np.nan, np.nan, 0
    else:
        child_ids = raw_children

    if not isinstance(child_ids, list) or len(child_ids) == 0:
        return np.nan, np.nan, 0

    parent_time = row["meta_date"]
    delays = []

    for cid in child_ids:
        if cid in id_to_date:
            child_time = id_to_date[cid]

            # FIX 1: Drop missing time info placeholders (exactly midnight)
            if child_time.hour == 0 and child_time.minute == 0 and child_time.second == 0:
                continue

            # FIX 2: If child is in 2019 and parent is in 2020, shift child to 2020
            # if child_time.year == 2019 and parent_time.year == 2020:
            #     child_time = child_time + pd.DateOffset(years=1)

            diff_seconds = (child_time - parent_time).total_seconds()
            
            # Catch-all safety filter to drop remaining edge-case negatives
            if diff_seconds >= 0:
                delays.append(diff_seconds)

    if not delays:
        return np.nan, np.nan, 0

    return np.mean(delays), np.median(delays), len(child_ids)


# 3. Use .apply() to process rows safely and efficiently
results = df.apply(calculate_row_delays, axis=1)

# 4. Unpack the results into your dataframe
df["mean_delay_seconds"] = [r[0] for r in results]
df["median_delay_seconds"] = [r[1] for r in results]
df["parsed_children_count"] = [r[2] for r in results]

# 5. Verify the final fix
negative_means = df[df["mean_delay_seconds"] < 0]
negative_medians = df[df["median_delay_seconds"] < 0]

print("Remaining negative rows (Means):", len(negative_means))
print("Remaining negative rows (Medians):", len(negative_medians))

print("rows in:", len(pd.read_csv("byproduct_data/families_with_children_stats.csv")))
print("rows out:", len(df))

# Save the cleaned dataset
df.to_csv("reply_time.csv", index=False)
