import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

DATA_DIR = "data/roberta/tokenized_chunks-stage2"
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_DIR = "saved_models/stage2_roberta"

BATCH_SIZE = 32
MAX_SAMPLES_DEBUG = None

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Loading test data from: {TEST_DIR}")
print(f"[INFO] Loading model from: {MODEL_DIR}")


def load_chunk(file_path):
    """
    Load a tokenized .pt chunk as TensorDataset.
    Compatible with PyTorch 2.6+ (weights_only issue).
    """
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


def softmax_numpy(logits_np):
    """Stable softmax for numpy array of shape (N, num_classes)."""
    z = logits_np - np.max(logits_np, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)


print("\n[INFO] Running evaluation on test set...")
test_loader = get_test_loader()

all_labels = []
all_preds = []
all_logits = []

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
        all_logits.append(logits.cpu().numpy())

        if MAX_SAMPLES_DEBUG is not None and len(all_labels) >= MAX_SAMPLES_DEBUG:
            print(f"[INFO] Reached MAX_SAMPLES_DEBUG={MAX_SAMPLES_DEBUG}, stopping early.")
            break

all_logits = np.concatenate(all_logits, axis=0)
y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_proba = softmax_numpy(all_logits)

print(f"[INFO] Collected {len(y_true)} test samples.")


print("\n" + "="*60)
print("OVERALL MODEL PERFORMANCE METRICS")
print("="*60)

accuracy = accuracy_score(y_true, y_pred)
print(f"\n1. Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

_, _, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
print(f"2. Macro-averaged F1: {f1_macro:.4f} ({f1_macro*100:.2f}%)")

_, _, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)
print(f"3. Weighted-averaged F1: {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

try:
    top1_acc = top_k_accuracy_score(
        y_true, y_proba, k=1, labels=list(range(num_labels))
    )
    top3_acc = top_k_accuracy_score(
        y_true, y_proba, k=min(3, num_labels), labels=list(range(num_labels))
    )
    print(f"4. Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"   Top-3 Accuracy: {top3_acc:.4f} ({top3_acc*100:.2f}%)")
except Exception as e:
    print(f"4. Top-k Accuracy: ERROR - {e}")
    top1_acc = None
    top3_acc = None

classes = list(range(num_labels))
y_true_bin = label_binarize(y_true, classes=classes)

try:
    roc_auc_macro = roc_auc_score(
        y_true_bin, y_proba, average="macro", multi_class="ovr"
    )
    roc_auc_weighted = roc_auc_score(
        y_true_bin, y_proba, average="weighted", multi_class="ovr"
    )
    print(f"5. Macro-averaged ROC-AUC (One-vs-Rest): {roc_auc_macro:.4f} ({roc_auc_macro*100:.2f}%)")
    print(f"6. Weighted-averaged ROC-AUC (One-vs-Rest): {roc_auc_weighted:.4f} ({roc_auc_weighted*100:.2f}%)")
    
    fpr_per_class = {}
    tpr_per_class = {}
    roc_auc_per_class = {}
    
    for i in range(num_labels):
        if (y_true == i).sum() > 0:
            fpr_per_class[i], tpr_per_class[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc_per_class[i] = auc(fpr_per_class[i], tpr_per_class[i])
    
    if len(fpr_per_class) > 0:
        all_fpr = np.unique(np.concatenate([fpr_per_class[i] for i in fpr_per_class.keys()]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in fpr_per_class.keys():
            mean_tpr += np.interp(all_fpr, fpr_per_class[i], tpr_per_class[i])
        mean_tpr /= len(fpr_per_class)
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
    else:
        fpr_macro = None
        tpr_macro = None
    
    plt.figure(figsize=(10, 8))
    
    if fpr_macro is not None:
        plt.plot(
            fpr_macro,
            tpr_macro,
            label=f"Macro-average ROC (AUC = {roc_auc_macro:.4f})",
            color="navy",
            linestyle="-",
            linewidth=2,
        )
    
    plt.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC = 0.50)", alpha=0.7)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(
        f"RoBERTa Stage 2 - ROC Curve (Macro-average)\n"
        f"Weighted ROC-AUC = {roc_auc_weighted:.4f}",
        fontsize=14
    )
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = "roberta_stage2_roc_curve_overall.png"
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[INFO] ROC curve plot saved to {roc_path}")
    print(f"      (Macro AUC = {roc_auc_macro:.4f}, Weighted AUC = {roc_auc_weighted:.4f})")
    
except Exception as e:
    print(f"5-6. ROC-AUC: ERROR - {e}")
    roc_auc_macro = None
    roc_auc_weighted = None


print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(f"{'Metric':<45} {'Value':<15}")
print("-"*60)
print(f"{'Accuracy':<45} {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'Macro-averaged F1':<45} {f1_macro:.4f} ({f1_macro*100:.2f}%)")
print(f"{'Weighted-averaged F1':<45} {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")
if top1_acc is not None:
    print(f"{'Top-1 Accuracy':<45} {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"{'Top-3 Accuracy':<45} {top3_acc:.4f} ({top3_acc*100:.2f}%)")
if roc_auc_macro is not None:
    print(f"{'Macro ROC-AUC (One-vs-Rest)':<45} {roc_auc_macro:.4f} ({roc_auc_macro*100:.2f}%)")
    print(f"{'Weighted ROC-AUC (One-vs-Rest)':<45} {roc_auc_weighted:.4f} ({roc_auc_weighted*100:.2f}%)")
print("="*60)

print("\n[INFO] Evaluation complete.")
