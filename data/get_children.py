import pandas as pd
import json

df = pd.read_csv("byproduct_data/threads_no_orphans.csv")

pairs = df.loc[df["expected_parent_path_id"].notna(), ["info_order", "expected_parent_path_id"]].copy()
pairs["expected_parent_path_id"] = pairs["expected_parent_path_id"].str.split(";;")
pairs = pairs.explode("expected_parent_path_id")
pairs["expected_parent_path_id"] = pairs["expected_parent_path_id"].str.split("->").str[1].str.strip()

children_map = pairs.groupby("expected_parent_path_id")["info_order"].apply(list)

df["children"] = df["info_order"].map(children_map)
df["children"] = df["children"].apply(lambda x: x if isinstance(x, list) else [])

df["children"] = df["children"].apply(json.dumps)
df.to_csv("full_families.csv", index=False)