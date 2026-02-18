"""
Memory-efficient tokenization for Stage 1 binary classification (~200k per class)
- Uses RoBERTa (roberta-base)
- Saves pre-tokenized chunks to disk (.pt) for later training
- Converts binary labels to integers: correct=1, wrong=0
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
import os
from tqdm import tqdm

MODEL_NAME = "roberta-base"
MAX_LEN = 256
CHUNK_SIZE = 5000
DATA_DIR = "data"
OUTPUT_DIR = "data/roberta/tokenized_chunks-stage1"

SPLITS = {
    "train": "stage1_train.csv",
    "val": "stage1_val.csv",
    "test": "stage1_test.csv"
}

def tokenize_split(split_name, csv_file):
    print(f"[INFO] Processing split: {split_name}")
    split_output_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df_iter = pd.read_csv(csv_file, chunksize=CHUNK_SIZE)
    chunk_count = 0

    for df_chunk in tqdm(df_iter, desc=f"Tokenizing {split_name} chunks"):
        chunk_count += 1

        if df_chunk["binary_label"].dtype == object:
            df_chunk["binary_label"] = df_chunk["binary_label"].map({"correct": 1, "wrong": 0})

        texts = df_chunk["line_content"].astype(str).tolist()
        labels = df_chunk["binary_label"].astype(int).tolist()

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

if __name__ == "__main__":
    for split, csv_name in SPLITS.items():
        csv_path = os.path.join(DATA_DIR, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"[ERROR] CSV file not found: {csv_path}")
        tokenize_split(split, csv_path)
    print("[INFO] All splits tokenized successfully!")
