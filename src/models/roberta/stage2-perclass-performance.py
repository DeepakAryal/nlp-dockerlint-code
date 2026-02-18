import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import classification_report
from joblib import load
import pandas as pd

MODEL_TYPE = "roberta"

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA_DIR = f"data/{MODEL_TYPE}/tokenized_chunks-stage2"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = f"saved_models/stage2_{MODEL_TYPE}"

BATCH_SIZE = 32
MAX_SAMPLES_DEBUG = None

print(f"[INFO] Analyzing: {MODEL_TYPE.upper()}")
print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading test data from: {TEST_DIR}")
print(f"[INFO] Loading model from: {MODEL_DIR}")


def load_chunk(file_path):
    """Load a tokenized .pt chunk as TensorDataset."""
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    return TensorDataset(
        data["input_ids"],
        data["attention_mask"],
        data["labels"],
    )


def get_test_loader():
    chunk_files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pt")))
    if len(chunk_files) == 0:
        raise RuntimeError(f"[ERROR] No chunk files found in {TEST_DIR}")

    print(f"[INFO] Found {len(chunk_files)} test chunk(s).")
    datasets = [load_chunk(f) for f in chunk_files]
    concat_dataset = ConcatDataset(datasets)

    loader = DataLoader(
        concat_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return loader


print("\n[INFO] Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()

num_labels = model.config.num_labels
print(f"[INFO] Model has {num_labels} classes.")


print("\n[INFO] Running evaluation on test set...")
test_loader = get_test_loader()

all_labels = []
all_preds = []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader, start=1):
        input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        if MAX_SAMPLES_DEBUG is not None and len(all_labels) >= MAX_SAMPLES_DEBUG:
            print(f"[INFO] Reached MAX_SAMPLES_DEBUG={MAX_SAMPLES_DEBUG}, stopping early.")
            break

y_true = np.array(all_labels)
y_pred = np.array(all_preds)

print(f"[INFO] Collected {len(y_true)} test samples.")


ENCODER_PATH = os.path.join(DATA_DIR, "label_encoder.joblib")
rule_names = None

if os.path.exists(ENCODER_PATH):
    print("\n[INFO] Loading label encoder to get rule names...")
    label_encoder = load(ENCODER_PATH)
    rule_names = label_encoder.inverse_transform(list(range(num_labels)))
else:
    print("[WARN] Label encoder not found, using generic class names.")
    rule_names = [f"CLASS_{i}" for i in range(num_labels)]


print("\n[INFO] Computing per-class metrics...")
report = classification_report(
    y_true,
    y_pred,
    target_names=[f"{i}: {rule}" for i, rule in enumerate(rule_names)],
    output_dict=True,
    zero_division=0,
)

rows = []
for idx in range(num_labels):
    key = f"{idx}: {rule_names[idx]}"
    if key in report:
        metrics = report[key]
        rows.append({
            "Rule_ID": idx,
            "Rule_Name": rule_names[idx],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1_Score": metrics["f1-score"],
            "Support": int(metrics["support"]),
        })

df = pd.DataFrame(rows)
df_sorted = df.sort_values("F1_Score", ascending=False)


print("\n" + "="*90)
print(f"TOP 10 BEST PERFORMING RULES (by F1 Score) - {MODEL_TYPE.upper()}")
print("="*90)
top10_best = df_sorted.head(10)
print(top10_best.to_string(index=False))

print("\n" + "="*90)
print(f"TOP 10 WORST PERFORMING RULES (by F1 Score) - {MODEL_TYPE.upper()}")
print("="*90)
top10_worst = df_sorted.tail(10)
print(top10_worst.to_string(index=False))

print("\n" + "="*90)
print("SUMMARY STATISTICS")
print("="*90)
print(f"Total number of rules: {len(df)}")
print(f"Average F1 Score: {df['F1_Score'].mean():.4f}")
print(f"Median F1 Score: {df['F1_Score'].median():.4f}")
print(f"Best F1 Score: {df['F1_Score'].max():.4f} (Rule: {df_sorted.iloc[0]['Rule_Name']})")
print(f"Worst F1 Score: {df['F1_Score'].min():.4f} (Rule: {df_sorted.iloc[-1]['Rule_Name']})")
print(f"Rules with F1 > 0.8: {(df['F1_Score'] > 0.8).sum()}")
print(f"Rules with F1 < 0.5: {(df['F1_Score'] < 0.5).sum()}")


csv_path = f"{MODEL_TYPE}_stage2_per_class_metrics.csv"
df_sorted.to_csv(csv_path, index=False)
print(f"\n[INFO] Full per-class metrics saved to {csv_path}")

print("\n[INFO] Analysis complete.")
