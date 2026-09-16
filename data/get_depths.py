import pandas as pd

df_f = pd.read_csv("byproduct_data/reply_time.csv")

# 1. Clean the thread ID
df_f["info_thread.id"] = df_f["info_thread.id"].fillna("unknown").astype(str)

# 2. Re-compute comment_depth accurately from info_order path length
df_f["comment_depth"] = df_f["info_order"].astype(str).str.split(",").apply(len)

# 3. Overwrite max_thread_depth cleanly
df_f["max_thread_depth"] = df_f.groupby("info_thread.id")["comment_depth"].transform("max")

print(df_f[df_f["info_thread.id"].isin(["2", "2.0"])][["info_order", "comment_depth", "max_thread_depth"]])

# Relative depth: How deep is this comment relative to the thread's max depth? (Range: ~0.0 to 1.0)
df_f["relative_comment_depth"] = df_f["comment_depth"] / df_f["max_thread_depth"]

# Depth delta: How far below the maximum thread depth is this comment?
df_f["depth_distance_from_max"] = df_f["max_thread_depth"] - df_f["comment_depth"]

def get_true_max_descendant_depth(df):
    # Map each comment's path to the max depth among all comments that start with that path
    paths = df["info_order"].astype(str).tolist()
    depths = df["comment_depth"].to_numpy()
    
    max_descendant_depths = []
    
    # Fast prefix check
    for current_path, current_depth in zip(paths, depths):
        prefix = current_path.split("-")[0] + ","
        # Find all comments that are descendants of this path
        descendant_depths = [
            d for p, d in zip(paths, depths) if p.startswith(prefix)
        ]
        
        if descendant_depths:
            relative_depth = max(descendant_depths) - current_depth
        else:
            relative_depth = 0
            
        max_descendant_depths.append(relative_depth)
        
    return max_descendant_depths

df_f["max_reply_depth"] = get_true_max_descendant_depth(df_f)

max_depths = df_f.groupby("info_thread.id")["comment_depth"].max()
thread_toxicities = (df_f["toxic"] == 1).groupby(df_f["info_thread.id"]).any()

toxic_thread_lengths = max_depths[thread_toxicities]
non_toxic_thread_lengths = max_depths[~thread_toxicities]

toxic_depths = df_f[df_f["toxic"] == 1]["max_reply_depth"].dropna()
nontoxic_depths = df_f[df_f["toxic"] == 0]["max_reply_depth"].dropna()

print(f"Toxic Mean:     {toxic_thread_lengths.mean():.4f}  (N = {len(toxic_thread_lengths)})")
print(f"Non-Toxic Mean: {non_toxic_thread_lengths.mean():.4f}  (N = {len(non_toxic_thread_lengths)})")

print(f"Toxic Mean:     {toxic_depths.mean():.4f}  (N = {len(toxic_depths)})")
print(f"Non-Toxic Mean: {nontoxic_depths.mean():.4f}  (N = {len(nontoxic_depths)})")

print("rows in:", len(pd.read_csv("/hpc/data/users/zlotfi/REU_Research/CAD/data1/reply_time.csv")))
print("rows out:", len(df_f))

df_f.to_csv("data_with_depths.csv", index=False)
