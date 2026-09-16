import json
import re
import pandas as pd

# 1. Custom function to safely sum strings containing text or numbers
def calculate_safe_sum(text_value):
    if pd.isna(text_value):
        return 0
    digits = re.findall(r'\d+', str(text_value))
    return sum(int(num) for num in digits)

# 2. Load your DataFrame
df = pd.read_csv("CAD/data1/og2.csv") 

temp_lists = df['info_order'].str.split(',')
df['is_original_post'] = df['info_order'].str.contains('-post', na=False)
df['list_len'] = temp_lists.str.len()
df['list_sum'] = df['info_order'].apply(calculate_safe_sum)

# 3. Sort your DataFrame
# Sorting guarantees that parents are processed before their children
df_sorted = df.sort_values(
    by=['is_original_post', 'list_len', 'list_sum'], 
    ascending=[False, True, True]
)

# 4. Helper function to find parent info_order string
def get_parent_id(info_order_str, is_post):
    if is_post or ',' not in str(info_order_str):
        return None
    # '1,4,2' -> '1,4'
    return ','.join(str(info_order_str).split(',')[:-1])

# Convert DataFrame rows into dictionary items and initialize a 'replies' list for each
comment_pool = {}
root_posts = []

# Clean up dataframe into a dictionary structure
columns_to_drop = ['list_len', 'list_sum', 'is_original_post']
records = df_sorted.drop(columns=columns_to_drop, errors='ignore').to_dict(orient='records')

# Step A: Initialize the lookup pool with an added "replies" list container
for record in records:
    record['replies'] = []
    comment_pool[record['info_order']] = record

# Step B: Construct the tree hierarchy mapping children to parents
for record in records:
    parent_id = get_parent_id(record['info_order'], '-post' in str(record['info_order']))
    
    if parent_id is None:
        # It's a top-level original thread post
        root_posts.append(record)
    else:
        # Find the parent in our pool and append this comment as a reply
        if parent_id in comment_pool:
            comment_pool[parent_id]['replies'].append(record)
        else:
            # Fallback if a parent comment is somehow missing from the dataset
            root_posts.append(record)

# 5. Save the structural tree to your HPC cluster JSON path
output_path = 'CAD/data1/og2.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(root_posts, f, indent=2, ensure_ascii=False)
