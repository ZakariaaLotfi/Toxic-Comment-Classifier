import pandas as pd

# 1. Load your finished sentiment data
PATH = "byproduct_data/og_with_toxicity.csv"
print(f"Reading {PATH}...")
df = pd.read_csv(PATH)
og_len = len(df)
print(f"Initial row count: {og_len}")

# Only group strictly by the unique structural ID
strict_anchors = ["info_id"]

# 2. Aggregate the changing columns into clean lists
def aggregate_to_list(series):
    return series.dropna().tolist()

changing_cols = {
    "id" : "first",
    "annotation_Primary" : aggregate_to_list,
    "annotation_Secondary" : aggregate_to_list,
    "annotation_Target" : aggregate_to_list,
    "annotation_Target_top.level.category" : aggregate_to_list,
    "annotation_highlighted" : aggregate_to_list
}

# ==========================================
# FIX: Dynamic loop automatically assigns 'first' to your other 21 metadata 
# columns (including floats and text). This bypasses string/float mismatch bugs completely.
# ==========================================
for col in df.columns:
    if col not in strict_anchors and col not in changing_cols:
        changing_cols[col] = "first"

print("Merging rows and combining metadata...\n")
merged_df = df.groupby(strict_anchors, dropna=False).agg(changing_cols).reset_index()

# Put 'id' back at the front
all_columns = ['id'] + [col for col in merged_df.columns if col != 'id']
merged_df = merged_df[all_columns]
merged_len = len(merged_df)

print(f"Final deduplicated row count: {merged_len}")

print(f"Total rows in original dataset: {og_len}")

# 1. Count how many times each unique comment text appears
# (We use info_id here since it uniquely anchors the comment entity)
appearance_counts = df["info_id"].value_counts()

# 2. Get a breakdown of your duplicates
print("\n--- Duplicate Distribution ---")
distribution = appearance_counts.value_counts().sort_index()
for occurrences, unique_comment_count in distribution.items():
    print(f"Comments appearing exactly {occurrences} time(s): {unique_comment_count}")

# 3. Calculate the exact expected row loss
# For every duplicate group, we lose (occurrences - 1) rows
total_rows_to_lose = 0
for occurrences, unique_comment_count in distribution.items():
    rows_lost_per_group = occurrences - 1
    total_rows_to_lose += (rows_lost_per_group * unique_comment_count)

expected_final_rows = og_len - total_rows_to_lose

print("\n--- Final Verification Math ---")
print(f"Initial Row Count:       {og_len}")
print(f"Expected Rows to Lose:   {total_rows_to_lose}")
print(f"Expected Final Rows:     {expected_final_rows}")
print(f"Resulted Final Rows:     {merged_len}")
print("-" * 31)

OUTPUT_PATH = "merged.csv"
merged_df.to_csv(OUTPUT_PATH, index=False)#, quoting=csv.QUOTE_NONNUMERIC)
print(f"Done! Saved merged data to {OUTPUT_PATH}")