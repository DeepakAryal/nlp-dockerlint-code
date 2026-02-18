"""
CodeBERT chunked training with progress bars (Stage 1 binary classification)
- Uses pre-tokenized .pt chunks (~200K per class)
- Shows per-chunk and per-batch progress with ETA
- Checkpoints model + optimizer + scheduler + tokenizer + history
- Resumes from last checkpoint if present
- Optimized for Mac (M2 GPU / 32GB RAM)
"""

import os
import json
from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

MODEL_NAME = "microsoft/codebert-base"
MODEL_SAVE_DIR = "saved_models/stage1_codebert"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

CHUNKS_DIR = "data/codebert/tokenized_chunks-stage1/train"
VAL_CHUNKS_DIR = "data/codebert/tokenized_chunks-stage1/val"

MAX_LEN = 256
BATCH_SIZE = 16
GRAD_ACCUM_STEPS = 2
EPOCHS = 2
LR = 2e-5
MAX_GRAD_NORM = 1.0
NUM_WORKERS = 0

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Device: {DEVICE}")

LAST_CKPT_DIR = os.path.join(MODEL_SAVE_DIR, "last_checkpoint")
HISTORY_FILE = os.path.join(MODEL_SAVE_DIR, "history.json")
META_FILE = os.path.join(LAST_CKPT_DIR, "meta.json")

def list_chunks(chunks_dir):
    p = Path(chunks_dir)
    if not p.exists():
        raise FileNotFoundError(f"Chunks dir not found: {chunks_dir}")
    return sorted([str(x) for x in p.iterdir() if x.suffix == ".pt"])

def load_chunk_dataset(chunk_file):
    data = torch.load(chunk_file, map_location="cpu", weights_only=False)
    return TensorDataset(data["input_ids"], data["attention_mask"], data["labels"])

def save_checkpoint(epoch, model, optimizer, scheduler, tokenizer, history):
    ckpt_dir = os.path.join(MODEL_SAVE_DIR, f"checkpoint_epoch{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "pytorch_model.bin"))
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }, os.path.join(ckpt_dir, "optim_sched.pt"))
    tokenizer.save_pretrained(ckpt_dir)

    os.makedirs(LAST_CKPT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(LAST_CKPT_DIR, "pytorch_model.bin"))
    torch.save({
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None
    }, os.path.join(LAST_CKPT_DIR, "optim_sched.pt"))
    tokenizer.save_pretrained(LAST_CKPT_DIR)

    with open(META_FILE, "w") as f:
        json.dump({"last_epoch": epoch}, f)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Checkpoint saved: epoch {epoch}")

def try_load_checkpoint(model, optimizer=None, scheduler=None):
    start_epoch = 0
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": []}
    if os.path.exists(LAST_CKPT_DIR):
        model_path = os.path.join(LAST_CKPT_DIR, "pytorch_model.bin")
        optim_path = os.path.join(LAST_CKPT_DIR, "optim_sched.pt")
        if os.path.exists(model_path):
            print(f"[INFO] Loading model from {LAST_CKPT_DIR}")
            map_location = DEVICE if DEVICE != "cuda" else None
            model.load_state_dict(torch.load(model_path, map_location=map_location))
        if optimizer and os.path.exists(optim_path):
            od = torch.load(optim_path, map_location="cpu")
            if "optimizer_state_dict" in od:
                optimizer.load_state_dict(od["optimizer_state_dict"])
            if scheduler and od.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(od["scheduler_state_dict"])
        if os.path.exists(META_FILE):
            with open(META_FILE, "r") as f:
                meta = json.load(f)
            start_epoch = meta.get("last_epoch", 0)
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
    return start_epoch, history

def compute_total_steps(chunk_files, batch_size):
    total_examples = sum(
        int(torch.load(f, map_location="cpu", weights_only=False)["labels"].shape[0])
        for f in tqdm(chunk_files, desc="Counting examples", ncols=100)
    )
    total_batches = math.ceil(total_examples / batch_size)
    return total_examples, total_batches

print("[INFO] Loading model and tokenizer...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR)

train_chunks = list_chunks(CHUNKS_DIR)
val_chunks = list_chunks(VAL_CHUNKS_DIR)
total_examples_train, total_batches_train = compute_total_steps(train_chunks, BATCH_SIZE)
effective_batches_per_epoch = math.ceil(total_batches_train / GRAD_ACCUM_STEPS)
total_training_steps = effective_batches_per_epoch * EPOCHS
warmup_steps = max(1, int(0.03 * total_training_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps)

start_epoch, history = try_load_checkpoint(model, optimizer, scheduler)
if start_epoch > 0:
    print(f"[INFO] Resuming training from epoch {start_epoch+1}")
else:
    print("[INFO] Starting training from scratch")

loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(start_epoch, EPOCHS):
    print(f"\n[INFO] ===== Epoch {epoch+1}/{EPOCHS} =====")
    model.train()
    running_loss, step_count, samples_seen = 0.0, 0, 0

    for chunk_idx, chunk_file in enumerate(train_chunks, start=1):
        dataset = load_chunk_dataset(chunk_file)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        n_batches = len(loader)

        with tqdm(loader, desc=f"Chunk {chunk_idx}/{len(train_chunks)}", ncols=120) as pbar:
            for batch_idx, batch in enumerate(pbar, start=1):
                input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels) / GRAD_ACCUM_STEPS
                loss.backward()

                running_loss += loss.item() * GRAD_ACCUM_STEPS
                samples_seen += input_ids.size(0)

                if (samples_seen // BATCH_SIZE) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    optimizer.zero_grad()
                    step_count += 1

                pbar.set_postfix({
                    "batch": f"{batch_idx}/{n_batches}",
                    "loss": f"{running_loss/(samples_seen/BATCH_SIZE):.4f}"
                })

        del dataset, loader
        torch.cuda.empty_cache() if DEVICE=="cuda" else None

    avg_train_loss = running_loss / max(1, samples_seen / BATCH_SIZE)
    print(f"[INFO] Epoch {epoch+1} avg train loss: {avg_train_loss:.4f}")

    model.eval()
    all_preds, all_labels = [], []
    val_loss, val_samples = 0.0, 0

    for chunk_file in val_chunks:
        dataset = load_chunk_dataset(chunk_file)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        with torch.inference_mode():
            for batch in tqdm(loader, desc="Validation", ncols=120):
                input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs.logits, labels)
                val_loss += loss.item() * input_ids.size(0)
                val_samples += input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        del dataset, loader
        torch.cuda.empty_cache() if DEVICE=="cuda" else None

    avg_val_loss = val_loss / max(1, val_samples)
    val_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", zero_division=0)
    print(f"[INFO] Epoch {epoch+1} Validation loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f}")

    history_row = {
        "epoch": epoch+1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_acc": val_acc,
        "val_precision": precision,
        "val_recall": recall,
        "val_f1": f1
    }
    for k,v in history_row.items():
        history.setdefault(k, []).append(v)
    save_checkpoint(epoch+1, model, optimizer, scheduler, tokenizer, history)

print("[INFO] Training complete!")
