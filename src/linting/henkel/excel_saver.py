import pandas as pd
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

class ExcelSaver:
    def __init__(self, output_file="dockerfile_analysis.xlsx"):
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
        Save all records to Excel file
        """
        if not self.records:
            print("[WARN] No records to save")
            return
        
        df = pd.DataFrame(self.records)
        
        column_order = ['source', 'file_id', 'line_number', 'line_content', 'label', 'label_rule', 'label_message']
        df = df[column_order]
        
        try:
            df.to_excel(self.output_file, index=False, engine='openpyxl')
            print(f"[SUCCESS] Saved {len(self.records)} records to {self.output_file}")
            
        except Exception as e:
            print(f"[WARN] Failed with openpyxl: {e}")
            try:
                df.to_excel(self.output_file, index=False, engine='xlsxwriter')
                print(f"[SUCCESS] Saved {len(self.records)} records to {self.output_file} (using xlsxwriter)")
            except Exception as e2:
                print(f"[ERROR] Failed with xlsxwriter: {e2}")
                try:
                    csv_file = self.output_file.replace('.xlsx', '.csv')
                    df.to_csv(csv_file, index=False)
                    print(f"[SUCCESS] Saved {len(self.records)} records to {csv_file} (CSV format)")
                except Exception as e3:
                    print(f"[ERROR] Failed to save in any format: {e3}")
                    return
        
        print(f"\n[SUMMARY] Analysis complete:")
        print(f"  Total lines analyzed: {len(self.records)}")
        print(f"  Correct lines: {len(df[df['label'] == 'correct'])}")
        print(f"  Wrong lines: {len(df[df['label'] == 'wrong'])}")
        print(f"  Files processed: {df['file_id'].nunique()}")
        
        print(f"\n[SAMPLE] First few results:")
        print(df.head(10).to_string(index=False))
