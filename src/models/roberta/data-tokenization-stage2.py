"""
Memory-efficient tokenization for Stage 2 multi-class classification (rule prediction)
- Uses RoBERTa (roberta-base)
- Saves pre-tokenized chunks to disk (.pt) for later training
- Converts label_rule to integers using LabelEncoder
- Optimized for GPU/CPU
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import joblib

MODEL_NAME = "roberta-base"
MAX_LEN = 256
CHUNK_SIZE = 5000
DATA_DIR = "data"
OUTPUT_DIR = "data/roberta/tokenized_chunks-stage2"
RANDOM_STATE = 42

SPLITS = {
    "train": "stage2_train.csv",
    "val": "stage2_val.csv",
    "test": "stage2_test.csv"
}

ENCODER_PATH = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

def tokenize_split(split_name, csv_file, label_encoder=None):
    """
    Tokenize a CSV split in chunks and save each chunk as .pt
    Returns label_encoder (fitted on train split)
    """
    print(f"[INFO] Processing split: {split_name}")
    split_output_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df_iter = pd.read_csv(csv_file, chunksize=CHUNK_SIZE)
    chunk_count = 0

    for df_chunk in tqdm(df_iter, desc=f"Tokenizing {split_name} chunks"):
        chunk_count += 1

        texts = df_chunk["line_content"].astype(str).tolist()
        rules = df_chunk["label_rule"].astype(str).tolist()

        if split_name == "train" and label_encoder is None:
            label_encoder = LabelEncoder()
            label_encoder.fit(rules)

        labels = label_encoder.transform(rules)

        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        dataset = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        chunk_file = os.path.join(split_output_dir, f"{split_name}_chunk_{chunk_count}.pt")
        torch.save(dataset, chunk_file, pickle_protocol=4)
        print(f"[INFO] Saved chunk {chunk_count} ({len(df_chunk)} rows) to {chunk_file}")

    return label_encoder

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    label_encoder = None

    for split, csv_name in SPLITS.items():
        csv_path = os.path.join(DATA_DIR, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file not found: {csv_path}")

        label_encoder = tokenize_split(split, csv_path, label_encoder)

    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"[INFO] Label encoder saved to {ENCODER_PATH}")
    print("[INFO] All splits tokenized successfully!")
