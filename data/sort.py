import pandas as pd

df = pd.read_csv("byproduct_data/merged.csv")

# ==========================================
# Re-apply your original Thread Sorting Logic
# ==========================================
print("Re-sorting rows back to contextual thread order...")

def reddit_thread_key(series):
    def parse_value(s):
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
                return (-1, 3, [s]) 
            
    return series.apply(parse_value)

# Apply sorting on the 'info_order' column of the merged dataframe
df = df.sort_values(by='info_order', key=reddit_thread_key)

df.to_csv("sorted.csv", index=False)