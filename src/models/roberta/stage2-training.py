"""
Stage 2: Train RoBERTa for Dockerfile rule prediction
- Target: label_rule
- Loads tokenized chunks (.pt) from Stage 2 tokenization
- Memory-efficient DataLoader with chunking
- Resume from last checkpoint
- Save model + tokenizer after training
- Shows progress with tqdm
"""

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_scheduler
import torch.nn as nn
import torch.optim as optim
import os
import glob
from sklearn.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 5e-5

DATA_DIR = "data/roberta/tokenized_chunks-stage2"
OUTPUT_DIR = "saved_models/stage2_roberta"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "last_checkpoint")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_chunk(file_path):
    """
    Load a tokenized .pt chunk as TensorDataset
    Fixes PyTorch 2.6+ weights_only issue
    """
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    return torch.utils.data.TensorDataset(
        data["input_ids"],
        data["attention_mask"],
        data["labels"]
    )

def get_loader(split):
    """
    Load all chunks for a split (train/val/test) and return DataLoader
    """
    split_dir = os.path.join(DATA_DIR, split)
    chunk_files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
    datasets = [load_chunk(f) for f in chunk_files]
    concat_dataset = ConcatDataset(datasets)
    loader = DataLoader(concat_dataset, batch_size=BATCH_SIZE, shuffle=(split=="train"))
    return loader

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_csv_path = "data/stage2_train.csv"
import pandas as pd
train_df = pd.read_csv(train_csv_path)
unique_rules = train_df["label_rule"].nunique()
print(f"[INFO] Number of label_rule classes: {unique_rules}")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=unique_rules
)
model.to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_loader = get_loader("train")
num_training_steps = EPOCHS * len(train_loader)
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
criterion = nn.CrossEntropyLoss()

checkpoint_path = os.path.join(CHECKPOINT_DIR, "pytorch_model.bin")
start_epoch = 0
if os.path.exists(checkpoint_path):
    print(f"[INFO] Resuming training from checkpoint {checkpoint_path} ...")
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)

val_loader = get_loader("val")

for epoch in range(start_epoch, EPOCHS):
    print(f"\n[INFO] Starting epoch {epoch+1}/{EPOCHS}")
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix({"loss": loss.item()})

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    print(f"[INFO] Epoch {epoch+1} train loss: {epoch_loss/len(train_loader):.4f}, acc: {acc:.4f}, f1: {f1:.4f}")

    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation", leave=False):
            input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")
    print(f"[INFO] Epoch {epoch+1} val loss: {val_loss/len(val_loader):.4f}, acc: {val_acc:.4f}, f1: {val_f1:.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"[INFO] Checkpoint saved to {checkpoint_path}")

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"[INFO] Stage 2 training completed. Model + tokenizer saved to {OUTPUT_DIR}")
