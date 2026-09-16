import csv
import pandas as pd

# 1. Read the CSV, skipping the few broken lines
df = pd.read_csv("og/og.csv")

# Reset the index to ensure no alignment mismatches after skipping lines
#df = df.reset_index(drop=True)

# 2. Explicit Map Approach (Safer than inverting)
# We explicitly define the 0s, and default everything else to 1
mapping = {
    "Neutral": 0,
    "CounterSpeech": 0,
    "IdentityDirectedAbuse": 1,
    "PersonDirectedAbuse": 1,
    "Slur": 1,
    "AffiliationDirectedAbuse": 1,
}

# .map() replaces the string with the 0 or 1. 
# .fillna(1) ensures if there's an unexpected blank/NaN, it defaults to toxic (or 0 if you prefer)
df["toxic"] = df["annotation_Primary"].map(mapping)#.fillna(2).astype(int)

# 3. Save directly to the new CSV
df.to_csv("og_with_toxicity.csv", index=False)#, quoting=csv.QUOTE_NONNUMERIC)