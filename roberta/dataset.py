import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def prepare_numerical_features(df, feature_cols, scaler=None):
    if not feature_cols:
        return None, None
    
    # Fill missing NaNs (e.g. comments with no children or ponos stats) with 0.0
    features = df[feature_cols].copy().fillna(0.0).values
    
    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
        
    return torch.tensor(features, dtype=torch.float32), scaler


class ContextualToxicityDataset(Dataset):
    def __init__(self, texts, labels, extra_features, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.extra_features = extra_features
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32)
        }
        
        if self.extra_features is not None:
            item["extra_features"] = self.extra_features[idx]
            
        return item