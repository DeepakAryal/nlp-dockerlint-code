import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

class ExcelSaver:
    def __init__(self, output_file="dockerfile_analysis.csv"):
        self.output_file = output_file
        self.records = []

    def add_records(self, records):
        """
        Add multiple records to the dataset
        """
        self.records.extend(records)
        print(f"[INFO] Added {len(records)} records. Total: {len(self.records)}")

    def save(self):
        """
        Save all records to CSV file.
        Appends if file already exists.
        """
        if not self.records:
            print("[WARN] No records to save")
            return

        df_new = pd.DataFrame(self.records)
        column_order = ['source', 'file_id', 'line_number', 'line_content', 'label', 'label_rule', 'label_message']
        df_new = df_new[column_order]

        try:
            df_new.to_csv(
                self.output_file,
                mode='a',
                index=False,
                header=not os.path.exists(self.output_file),
                encoding='utf-8'
            )
            print(f"[SUCCESS] Appended {len(df_new)} records to {self.output_file}")
        except Exception as e:
            print(f"[ERROR] Failed to save CSV: {e}")
            return

        total_lines = 0
        try:
            df_total = pd.read_csv(self.output_file)
            total_lines = len(df_total)
            correct_lines = len(df_total[df_total['label'] == 'correct'])
            wrong_lines = len(df_total[df_total['label'] == 'wrong'])
            files_processed = df_total['file_id'].nunique()
        except Exception as e:
            print(f"[WARN] Could not read CSV for summary: {e}")
            correct_lines = wrong_lines = files_processed = 0

        print(f"\n[SUMMARY] Analysis complete:")
        print(f"  Total lines analyzed: {total_lines}")
        print(f"  Correct lines: {correct_lines}")
        print(f"  Wrong lines: {wrong_lines}")
        print(f"  Files processed: {files_processed}")

        print(f"\n[SAMPLE] First few results:")
        print(df_new.head(10).to_string(index=False))
