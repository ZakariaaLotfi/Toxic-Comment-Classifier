import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt


def plot_learning_curves(history, save_path="learning_curves.png"):
    """
    Plots Train Loss, Dev Loss (if tracked), Dev F1, and Dev AUC over epochs.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Loss ---
    ax1.plot(epochs, history["train_loss"], 'o-', label='Train Loss', color='crimson')
    if "dev_loss" in history and history["dev_loss"]:
        ax1.plot(epochs, history["dev_loss"], 's--', label='Dev Loss', color='darkorange')
    ax1.set_title('Loss over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # --- Plot 2: Evaluation Metrics ---
    ax2.plot(epochs, history["dev_f1"], 'o-', label='Dev F1 Score', color='navy')
    if "dev_auc" in history and history["dev_auc"]:
        ax2.plot(epochs, history["dev_auc"], '^--', label='Dev ROC-AUC', color='teal')
    ax2.set_title('Validation Performance over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to: {save_path}")


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, criterion=None):
    model.train()
    total_loss = 0.0
    
    # Default to standard BCEWithLogitsLoss if custom weighted criterion is not passed
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()

    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        extra_features = batch.get("extra_features")
        if extra_features is not None:
            extra_features = extra_features.to(device)

        with torch.amp.autocast('cuda'):
            logits = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, threshold=0.5, return_probs=False):
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            extra_features = batch.get("extra_features")
            if extra_features is not None:
                extra_features = extra_features.to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask, extra_features=extra_features)
            
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_probs = torch.sigmoid(torch.tensor(all_logits)).numpy()
    all_labels = np.array(all_labels)
    preds = (all_probs >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, average="binary", zero_division=0),
        "f1_macro": f1_score(all_labels, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(all_labels, preds, average="micro", zero_division=0),
        "precision": precision_score(all_labels, preds, average="binary", zero_division=0),
        "recall": recall_score(all_labels, preds, average="binary", zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }

    if return_probs:
        metrics["probs"] = all_probs

    return metrics


def save_checkpoint(model, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)


def save_results(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([metrics])
    
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)