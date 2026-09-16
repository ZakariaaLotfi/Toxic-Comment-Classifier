import pandas as pd

# 1. Load your dataset
df = pd.read_csv("byproduct_data/threads_with_sentiments.csv")

# Ensure columns are clean strings
df["info_order"] = df["info_order"].astype(str).str.strip()
df["info_thread.id"] = df["info_thread.id"].astype(str).str.strip()

# Create a temporary unique index to track rows reliably across operations
df["temp_row_id"] = range(len(df))

# Keep a full backup copy of the original dataframe to extract deleted rows later
original_df = df.copy()

thread_error_counts = {}
sub_reply_orphan_counts = {}

broken_thread_ids = set()
orphan_rows = pd.DataFrame()
non_orphan_rows = pd.DataFrame()
replies_df_m = pd.DataFrame()
complete_pairs = 0
incomplete_pairs = 0
missing_title = 0
missing_post = 0
orphan_threads = 0
complete_threads = 0

def orphan_and_title_post_pair_stats(df, complete_threads, orphan_threads):
    def get_parent_path_id(row):
        path = row["info_order"]
        thread = row["info_thread.id"]

        if "-" in path and "," not in path:
            return None

        parts = [p.strip() for p in path.split(",")]
        base_root = parts[0].split("-")[0] if "-" in parts[0] else parts[0]

        if len(parts) == 2:
            expected_post_path = f"{thread} -> {base_root}-post"
            expected_title_path = f"{thread} -> {base_root}-title"

            if expected_post_path in existing_paths and expected_title_path in existing_paths:
                return f"{expected_title_path};;{expected_post_path}"
            elif expected_post_path in existing_paths:
                return expected_post_path
            elif expected_title_path in existing_paths:
                return expected_title_path
            return expected_post_path
        
        parent_path = ", ".join(parts[:-1])
        return f"{thread} -> {parent_path}"
    
    complete_pairs = 0
    incomplete_pairs = 0
    missing_post = 0
    missing_title = 0

    # Construct current path realities
    df["thread_path_id"] = df["info_thread.id"] + " -> " + df["info_order"]
    existing_paths = set(df["thread_path_id"].unique())

    # Calculate expected parent IDs
    df["expected_parent_path_id"] = df.apply(get_parent_path_id, axis=1)

    # Isolate replies that structurally demand a parent
    replies_df = df[df["expected_parent_path_id"].notna()]

    def parent_exists(expected):
        if not isinstance(expected, str):
            return False
        return any(c in existing_paths for c in expected.split(";;"))

    exists_mask = replies_df["expected_parent_path_id"].apply(parent_exists)
    is_orphan_mask = ~exists_mask
    not_orphan_mask = exists_mask

    orphan_rows = replies_df[is_orphan_mask]
    non_orphan_rows = replies_df[not_orphan_mask]

    # Count how many sub-reply orphans are in each thread
    sub_reply_orphan_counts = orphan_rows["info_thread.id"].value_counts()

    # Merge the sub-reply orphan counts into our localized dictionary
    for t_id, count in sub_reply_orphan_counts.items():
        thread_error_counts[t_id] = thread_error_counts.get(t_id, 0) + count

    # Identify exactly which individual threads are broken
    broken_thread_ids = set(thread_error_counts.keys())

    # Isolate top-level rows (rows without commas in info_order)
    top_level_mask = ~df["info_order"].str.contains(",")
    top_level_df = df[top_level_mask].copy()
    top_level_df["base_id"] = top_level_df["info_order"].apply(
        lambda x: x.split("-")[0]
    )

    grouped_top = top_level_df.groupby(["info_thread.id", "base_id"])

    for (thread_id, base_id), group in grouped_top:
        suffixes = set(
            group["info_order"].apply(
                lambda x: x.split("-")[1] if "-" in x else "missing_suffix"
            )
        )
        has_title = "title" in suffixes
        has_post = "post" in suffixes

        if has_title and has_post:
            complete_pairs += 1
        else:
            incomplete_pairs += 1
        if has_title and not has_post: missing_post += 1
        elif not has_title and has_post: missing_title += 1

    print(f"-> Complete Top-Level Pairs:   {complete_pairs}")
    print(f"-> Incomplete Top-Level Pairs: {incomplete_pairs}")
    print(f"-> Incomplete Missing Title:   {missing_title}")
    print(f"-> Incomplete Missing Post:    {missing_post}\n")

    # ==========================================
    # PRINT IMPORTANT METRICS (FIXED TO USE CURRENT DF)
    # ==========================================
    total_threads = df["info_thread.id"].nunique()
    orphan_threads = len(broken_thread_ids)
    complete_threads = total_threads - orphan_threads

    print(f"-> Total Unique Threads Analyzed:  {total_threads}")
    print(f"-> Threads with Orphans/Holes:     {orphan_threads}")
    print(f"-> Complete Threads:               {complete_threads}")
    print(f"-> Total Direct Orphan Rows Found: {len(orphan_rows)}\n")
    
    # ==========================================
    # ANALYSIS: COMPLETELY MISSING TOP-LEVEL PAIRS & THEIR ORPHANS
    # ==========================================
    print("ANALYSIS: COMPLETELY MISSING TOP-LEVEL PAIRS & THEIR ORPHANS")

    # 1. Get a set of all top-level base IDs that actually exist in each thread
    existing_top_levels = set()
    for (thread_id, base_id), _ in grouped_top:
        existing_top_levels.add(f"{thread_id} -> {base_id}")

    # 2. Scan all replies to find what top-level base ID they expect to branch from
    expected_top_levels = set()
    for idx, row in replies_df.iterrows():
        path = row["info_order"]
        thread_id = row["info_thread.id"]
        
        parts = [p.strip() for p in path.split(",")]
        base_root = parts[0].split("-")[0] if "-" in parts[0] else parts[0]
        expected_top_levels.add(f"{thread_id} -> {base_root}")

    # 3. Find the difference: expected roots that do not exist at all
    completely_missing_top_pairs = expected_top_levels - existing_top_levels

    # 4. Count the amount of orphan rows caused by these completely missing pairs
    orphans_directly_replying_to_missing_roots = 0

    for idx, row in orphan_rows.iterrows():
        path = row["info_order"]
        thread_id = row["info_thread.id"]
        
        parts = [p.strip() for p in path.split(",")]
        
        # Only look at immediate direct replies
        if len(parts) == 2:
            base_root = parts[0].split("-")[0] if "-" in parts[0] else parts[0]
            orphan_root_identity = f"{thread_id} -> {base_root}"
            
            if orphan_root_identity in completely_missing_top_pairs:
                orphans_directly_replying_to_missing_roots += 1

    print(f"-> Orphans DIRECTLY replying to completely missing pairs: {orphans_directly_replying_to_missing_roots}")

    print(f"-> Unique top-level pairs completely missing (0 rows):    {len(completely_missing_top_pairs)}")

    # Cross-reference with total orphans for context
    total_orphans = len(orphan_rows)
    if total_orphans > 0:
        pct = (orphans_directly_replying_to_missing_roots / total_orphans) * 100
        print(f"-> Share of total orphans:                        {pct:.1f}%")

    print("\nREMOVING ORPHANS AND THEIR DESCENDANTS")

    # Full path IDs of every orphan (thread-scoped)
    orphan_path_ids = set(
        orphan_rows["info_thread.id"] + " -> " + orphan_rows["info_order"]
    )

    def is_orphan_or_descendant(row):
        row_path_id = row["thread_path_id"]

        # The orphan itself
        if row_path_id in orphan_path_ids:
            return True

        thread_id = row["info_thread.id"]
        path = row["info_order"]
        parts = [p.strip() for p in path.split(",")]

        # Check every ancestor prefix of this row's path against the orphan set.
        # If ANY ancestor node is an orphan, this row is a descendant of it
        # (this naturally covers descendants at any depth, not just direct children).
        for i in range(1, len(parts)):
            ancestor_path = ", ".join(parts[:i])
            ancestor_path_id = f"{thread_id} -> {ancestor_path}"
            if ancestor_path_id in orphan_path_ids:
                return True

        return False

    descendant_mask = df.apply(is_orphan_or_descendant, axis=1)
    removed_count = descendant_mask.sum()
    df_cleaned = df[~descendant_mask].copy()

    df[descendant_mask].to_csv("orphans_and_descendants.csv", index=False)
    df_cleaned.to_csv("threads_no_orphans.csv", index=False)

    print(f"-> Total rows removed (orphans + descendants): {removed_count}")
    print(f"-> Rows remaining after cleaning: {len(df_cleaned)}\n")

orphan_and_title_post_pair_stats(df, complete_threads, orphan_threads)