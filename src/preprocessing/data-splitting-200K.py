"""
Stage 1: Binary label reduced split (~200k per class)
- Reads balanced dataset (~801k per class)
- Reduces each class to ~200k
- Creates stratified train/val/test splits (~80/10/10)
- Saves CSVs in data folder
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

CSV_PATH = "data/dockerfile-analysis-balanced.csv"
TARGET_PER_CLASS = 200_000
RANDOM_STATE = 42
OUTPUT_DIR = "data"

def downsample_per_class(df, label_col, target_per_class):
    """Return dataframe with target_per_class rows per class, shuffled."""
    sampled_list = []
    for cls in df[label_col].unique():
        cls_df = df[df[label_col] == cls]
        sampled_list.append(cls_df.sample(n=target_per_class, random_state=RANDOM_STATE))
    df_down = pd.concat(sampled_list).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df_down

def save_splits(train, val, test, stage_name="stage1"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train.to_csv(os.path.join(OUTPUT_DIR, f"{stage_name}_train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, f"{stage_name}_val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, f"{stage_name}_test.csv"), index=False)
    print(f"[INFO] {stage_name} splits saved:")
    print(f" - train: {len(train)} rows")
    print(f" - val:   {len(val)} rows")
    print(f" - test:  {len(test)} rows")

def prepare_stage1_reduced():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    df = df.dropna(subset=['line_content'])

    df['binary_label'] = df['label'].apply(lambda x: 'correct' if str(x).strip().lower() == 'correct' else 'wrong')

    df_down = downsample_per_class(df, 'binary_label', TARGET_PER_CLASS)

    train_val, test = train_test_split(
        df_down, test_size=0.10, stratify=df_down['binary_label'], random_state=RANDOM_STATE
    )
    train, val = train_test_split(
        train_val, test_size=0.111111, stratify=train_val['binary_label'], random_state=RANDOM_STATE
    )

    save_splits(train, val, test, stage_name="stage1")

if __name__ == "__main__":
    prepare_stage1_reduced()
