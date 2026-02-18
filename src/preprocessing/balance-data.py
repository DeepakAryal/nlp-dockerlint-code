"""
Simple script to balance Stage 1 dataset for binary classification.
- Downsamples 'correct' label to match 'wrong' label (~801,687 rows each)
- Shuffles the dataset
- Saves the balanced dataset as CSV
"""

import pandas as pd

CSV_PATH = "data/dockerfile_analysis.csv"
OUTPUT_PATH = "data/dockerfile-analysis-balanced.csv"
RANDOM_STATE = 42

print(f"[INFO] Loading dataset from {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH, encoding='utf-8')

df = df.dropna(subset=['line_content'])
print(f"[INFO] Dataset loaded: {len(df)} rows after dropping missing content")

df['binary_label'] = df['label'].apply(
    lambda x: 'correct' if str(x).strip().lower() == 'correct' else 'wrong'
)
print("[INFO] Binary labels created")
print(df['binary_label'].value_counts())

df_wrong = df[df['binary_label'] == 'wrong']
df_correct = df[df['binary_label'] == 'correct']

n_wrong = len(df_wrong)
df_correct_down = df_correct.sample(n=n_wrong, random_state=RANDOM_STATE)

df_balanced = pd.concat([df_wrong, df_correct_down]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

df_balanced.to_csv(OUTPUT_PATH, index=False)
print(f"[INFO] Balanced dataset saved to {OUTPUT_PATH}")
print(f"[INFO] Total rows: {len(df_balanced)}")
print(df_balanced['binary_label'].value_counts())
