"""
Stage 2: Rule prediction data split
- Uses ORIGINAL dockerfile_analysis.csv
- Keeps ONLY misconfigured lines (binary_label = wrong)
- Predicts label_rule (multi-class)
- Carefully balances rules (not binary balancing)
- Stratified train/val/test split (80/10/10)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

CSV_PATH = "data/dockerfile_analysis.csv"
OUTPUT_DIR = "data"
RANDOM_STATE = 42

MIN_SAMPLES_PER_RULE = 200
MAX_SAMPLES_PER_RULE = 5000

print("[INFO] Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="utf-8")
df = df.dropna(subset=["line_content", "label_rule"])

df = df[df["label"].str.lower() != "correct"]
print(f"[INFO] Misconfigured samples: {len(df)}")

df = df[df["label_rule"].str.lower() != "none"]

rule_counts = df["label_rule"].value_counts()
print("[INFO] Rules before balancing:", len(rule_counts))

balanced_chunks = []

for rule, count in rule_counts.items():
    rule_df = df[df["label_rule"] == rule]

    if count < MIN_SAMPLES_PER_RULE:
        continue

    if count > MAX_SAMPLES_PER_RULE:
        rule_df = rule_df.sample(
            n=MAX_SAMPLES_PER_RULE,
            random_state=RANDOM_STATE
        )

    balanced_chunks.append(rule_df)

df_balanced = (
    pd.concat(balanced_chunks)
    .sample(frac=1, random_state=RANDOM_STATE)
    .reset_index(drop=True)
)

print(f"[INFO] Samples after rule balancing: {len(df_balanced)}")
print(f"[INFO] Number of rules kept: {df_balanced['label_rule'].nunique()}")

train_val, test = train_test_split(
    df_balanced,
    test_size=0.10,
    stratify=df_balanced["label_rule"],
    random_state=RANDOM_STATE
)

train, val = train_test_split(
    train_val,
    test_size=0.111111,
    stratify=train_val["label_rule"],
    random_state=RANDOM_STATE
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train.to_csv(os.path.join(OUTPUT_DIR, "stage2_train.csv"), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, "stage2_val.csv"), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, "stage2_test.csv"), index=False)

print("[INFO] Stage 2 splits saved:")
print(f" - Train: {len(train)}")
print(f" - Val:   {len(val)}")
print(f" - Test:  {len(test)}")
