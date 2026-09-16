import pandas as pd

df = pd.read_csv("byproduct_data/sorted.csv")

df = df[["info_order","id","info_id","annotation_Primary","annotation_Secondary","annotation_Target","annotation_Target_top.level.category",
         "annotation_highlighted","info_subreddit","info_subreddit_id","info_id.parent","info_id.link","info_thread.id","info_image.saved",
         "annotation_Context","meta_author","meta_created_utc","meta_date","meta_day","meta_permalink","split","subreddit_seen","meta_text","toxic"]]

df.to_csv("sorted.csv",index=False)