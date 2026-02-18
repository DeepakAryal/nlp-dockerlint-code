import pandas as pd
from collections import Counter

CSV_PATH = "data/dockerfile_analysis.csv"

df = pd.read_csv(CSV_PATH, encoding='utf-8')

# Stage 1: binary counts
df['binary_label'] = df['label'].apply(lambda x: 'correct' if str(x).strip().lower() == 'correct' else 'wrong')
binary_counts = df['binary_label'].value_counts()
print("[INFO] Stage 1 - binary label counts:")
print(binary_counts)
print(f"Total lines: {len(df)}\n")

# Stage 2: wrong lines and rule counts
df_wrong = df[df['binary_label'] == 'wrong']
rule_counts = df_wrong['label_rule'].value_counts()
print("[INFO] Stage 2 - label_rule counts (only wrong lines):")
print(rule_counts)
print(f"Total wrong lines: {len(df_wrong)}\n")

# Check class imbalance ratios
if len(binary_counts) == 2:
    ratio = binary_counts['correct'] / binary_counts['wrong']
    print(f"[INFO] Correct/Wrong ratio: {ratio:.2f}")

# Identify rare rules
MIN_SAMPLES = 20
rare_rules = rule_counts[rule_counts < MIN_SAMPLES]
print("[INFO] Rare rules with fewer than 20 examples:")
print(rare_rules)
