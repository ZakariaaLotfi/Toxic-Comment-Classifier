import pandas as pd

PATH = "byproduct_data/og_with_toxicity.csv"
print(f"Analyzing {PATH}...")
df = pd.read_csv(PATH)

initial_rows = len(df)
print(f"Total rows in original dataset: {initial_rows}")

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

expected_final_rows = initial_rows - total_rows_to_lose

print("\n--- Final Verification Math ---")
print(f"Initial Row Count:       {initial_rows}")
print(f"Expected Rows to Lose:   {total_rows_to_lose}")
print(f"Expected Final Rows:     {expected_final_rows}")
print("-" * 31)