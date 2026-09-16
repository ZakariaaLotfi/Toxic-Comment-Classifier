import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import RobertaTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score

from CAD.train.roberta.dataset import ContextualToxicityDataset, prepare_numerical_features
from CAD.train.roberta.model import RobertaContextualClassifier
from CAD.train.roberta.trainer import train_epoch, evaluate, save_checkpoint, save_results, plot_learning_curves

# Configuration
CSV_PATH = "../../data/data_with_depths.csv"
MODEL_PATH = "../../models/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b"
BATCH_SIZE = 64
EPOCHS = 8
LR = 2e-5
USE_SAMPLER = False

# Feature columns: All extra features combined
FEATURE_COLS = [
    # "sentiment_positive",
    # "sentiment_neutral",
    # "sentiment_negative",
    "practical_sentiment_score",
    # "top_sentiment_score",
    # "children_mean_sentiment",
    # "ponos",
    # "mean_delay_seconds",
    "max_thread_depth"
]

def find_optimal_threshold(dev_probs, dev_labels):
    best_thresh, best_f1 = 0.5, 0.0
    for thresh in np.arange(0.10, 0.90, 0.02):
        preds = (dev_probs >= thresh).astype(int)
        score = f1_score(dev_labels, preds, average="binary", zero_division=0)
        if score > best_f1:
            best_f1, best_thresh = score, thresh
    return best_thresh, best_f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing File 6 (Text + Sentiment + Depth) [SUBSET ONLY] on Device: {device}")
    
    # 1. Load Data & Filter ONLY rows with Children/PONOS metadata
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["children_mean_sentiment"].notna() & df["ponos"].notna()].reset_index(drop=True)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    print(f"Subset samples -> Train: {len(train_df)} | Dev: {len(dev_df)} | Test: {len(test_df)}")
    
    # 2. Prepare & Scale Numerical Features
    train_features, scaler = prepare_numerical_features(train_df, FEATURE_COLS)
    dev_features, _ = prepare_numerical_features(dev_df, FEATURE_COLS, scaler=scaler)
    test_features, _ = prepare_numerical_features(test_df, FEATURE_COLS, scaler=scaler)
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    
    train_dataset = ContextualToxicityDataset(train_df["meta_text"], train_df["toxic"], train_features, tokenizer)
    dev_dataset = ContextualToxicityDataset(dev_df["meta_text"], dev_df["toxic"], dev_features, tokenizer)
    test_dataset = ContextualToxicityDataset(test_df["meta_text"], test_df["toxic"], test_features, tokenizer)
    
    # 3. Dynamic Batch Sampling
    num_neg = (train_df["toxic"] == 0).sum()
    num_pos = (train_df["toxic"] == 1).sum()
    
    if USE_SAMPLER:
        class_weights = [1.0 / num_neg, 1.0 / num_pos]
        sample_weights = [class_weights[int(label)] for label in train_df["toxic"]]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Dynamic pos_weight for Loss Function
    pos_weight = torch.tensor([1.0] if USE_SAMPLER else [num_neg / num_pos], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model = RobertaContextualClassifier(model_path_or_name=MODEL_PATH, num_extra_features=len(FEATURE_COLS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    scaler_amp = torch.amp.GradScaler('cuda')
    
    history = {
        "train_loss": [],
        "dev_f1": [],
        "dev_auc": []
    }

    best_dev_f1 = 0.0

    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler_amp, device, criterion=criterion)
        dev_metrics = evaluate(model, dev_loader, device)
        
        # Record history
        history["train_loss"].append(train_loss)
        history["dev_f1"].append(dev_metrics["f1"])
        history["dev_auc"].append(dev_metrics["roc_auc"])
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Dev Acc: {dev_metrics['accuracy']:.4f} | Dev F1: {dev_metrics['f1']:.4f} | Dev AUC: {dev_metrics['roc_auc']:.4f}")
        
        # Checkpoint ONLY the best performing epoch
        if dev_metrics["f1"] > best_dev_f1:
            best_dev_f1 = dev_metrics["f1"]
            save_checkpoint(model, "checkpoints/best_model_6.pt")
            print(f"  --> Saved new best model checkpoint (Dev F1: {best_dev_f1:.4f})")

    # Generate and save the visual plot after training finishes
    plot_learning_curves(history, save_path="results/learning_curves_subset6.png")    
    
    # 5. Optimize Threshold on Dev Split
    dev_results = evaluate(model, dev_loader, device, return_probs=True) if "return_probs" in evaluate.__code__.co_varnames else evaluate(model, dev_loader, device)
    if isinstance(dev_results, dict) and "probs" in dev_results:
        optimal_thresh, best_dev_f1 = find_optimal_threshold(dev_results["probs"], dev_df["toxic"].values)
    else:
        optimal_thresh = 0.5
        best_dev_f1 = dev_metrics.get("f1", 0.0)

    print(f"\nOptimal Dev Decision Threshold: {optimal_thresh:.2f} (Dev Binary F1: {best_dev_f1:.4f})")

    # 6. Final Test Evaluation
    test_metrics = evaluate(model, test_loader, device, threshold=optimal_thresh) if "threshold" in evaluate.__code__.co_varnames else evaluate(model, test_loader, device)
    test_metrics["threshold"] = round(optimal_thresh, 2)
    test_metrics["experiment"] = "6_text_sentiment_depth_subset"
    
    print("\n--- Final Test Set Results (Subset) ---")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            
    save_results(test_metrics, "results/experiment_results6_subset.csv")
    save_checkpoint(model, "checkpoints/6_text_sentiment_depth_subset.pt")

if __name__ == "__main__":
    main()