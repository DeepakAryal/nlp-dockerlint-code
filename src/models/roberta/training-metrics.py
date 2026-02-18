import json
import os
import matplotlib.pyplot as plt

HISTORY_PATH = "saved_models/stage1_roberta/history.json"
OUTPUT_DIR = "saved_models/stage1_roberta/plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

epochs = history["epoch"]
train_loss = history["train_loss"]
val_loss = history["val_loss"]
val_f1 = history["val_f1"]
val_precision = history["val_precision"]
val_recall = history["val_recall"]

plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RoBERTa: Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"))
plt.close()

plt.figure()
plt.plot(epochs, val_f1, marker="o", label="Validation F1-score")
plt.xlabel("Epoch")
plt.ylabel("F1-score")
plt.title("RoBERTa: Validation F1-score over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "val_f1_curve.png"))
plt.close()

plt.figure()
plt.plot(epochs, val_precision, marker="o", label="Validation Precision")
plt.plot(epochs, val_recall, marker="o", label="Validation Recall")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("RoBERTa: Validation Precision and Recall over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "val_precision_recall_curve.png"))
plt.close()

print("✅ RoBERTa plots saved to:", OUTPUT_DIR)
