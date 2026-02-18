"""
Evaluate Stage 1 RoBERTa model for Dockerfile misconfiguration detection.
- Loads model from last_checkpoint/pytorch_model.bin (state_dict)
- Memory-efficient chunked evaluation
- Computes Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC
- Compatible with PyTorch 2.6+ (weights_only fix)
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification
import os
import glob
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

MODEL_DIR = "saved_models/stage1_roberta/last_checkpoint"
TOKENIZED_TEST_DIR = "data/roberta/tokenized_chunks-stage1/test"
BATCH_SIZE = 32

def load_chunk(file_path):
    """
    Load a tokenized .pt chunk as TensorDataset
    Fixes PyTorch 2.6+ weights_only issue by setting weights_only=False
    """
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    return TensorDataset(
        data["input_ids"],
        data["attention_mask"],
        data["labels"]
    )

print(f"[INFO] Loading RoBERTa model from {MODEL_DIR} ...")

model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=2
)

state_dict_path = os.path.join(MODEL_DIR, "pytorch_model.bin")
model.load_state_dict(torch.load(state_dict_path, map_location=DEVICE))

model.to(DEVICE)
model.eval()

print("[INFO] Model loaded successfully.")

all_labels = []
all_preds = []
all_probs = []

chunk_files = sorted(glob.glob(os.path.join(TOKENIZED_TEST_DIR, "*.pt")))

with torch.no_grad():
    for chunk_file in tqdm(chunk_files, desc="Evaluating chunks"):
        dataset = load_chunk(chunk_file)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        for batch in loader:
            input_ids, attention_mask, labels = [t.to(DEVICE) for t in batch]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

print("\nClassification Report:\n")
print(classification_report(
    all_labels,
    all_preds,
    target_names=["correct", "wrong"]
))

cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:\n", cm)

fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = auc(fpr, tpr)

print(f"\nROC AUC: {roc_auc:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Stage 1 RoBERTa ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
