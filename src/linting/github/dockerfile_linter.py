import subprocess
import json
import os
from pathlib import Path
import pandas as pd
import re

class HadolintLinter:
    def __init__(self):
        self.hadolint_path = "hadolint"
    
    def lint_file(self, file_path):
        """
        Lint a Dockerfile using Hadolint and return JSON output
        """
        try:
            result = subprocess.run([
                self.hadolint_path,
                "--format", "json",
                "--no-fail",
                str(file_path)
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 or result.stdout:
                try:
                    lint_results = json.loads(result.stdout.strip())
                    return lint_results
                except json.JSONDecodeError:
                    print(f"[WARN] Failed to parse JSON output for {file_path}")
                    return []
            else:
                print(f"[WARN] Hadolint failed for {file_path}: {result.stderr}")
                return []
                
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Hadolint timeout for {file_path}")
            return []
        except Exception as e:
            print(f"[ERROR] Hadolint error for {file_path}: {e}")
            return []
    
    def get_line_labels(self, file_path, lint_results):
        """
        Parse Dockerfile lines and label them based on Hadolint results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[ERROR] Failed to read {file_path}: {e}")
            return []
        
        line_issues = {}
        for issue in lint_results:
            line_num = issue.get('line', 0)
            if line_num not in line_issues:
                line_issues[line_num] = []
            line_issues[line_num].append(issue)
        
        labeled_lines = []
        file_id = Path(file_path).stem
        
        for line_num, line_content in enumerate(lines, 1):
            line_content = line_content.rstrip('\n\r')
            
            if not line_content.strip():
                continue
            
            if line_num in line_issues:
                for issue in line_issues[line_num]:
                    labeled_lines.append({
                        'source': 'github-api',
                        'file_id': file_id,
                        'line_number': line_num,
                        'line_content': line_content,
                        'label': 'wrong',
                        'label_rule': issue.get('code', 'unknown'),
                        'label_message': issue.get('message', 'No message')
                    })
            else:
                labeled_lines.append({
                    'source': 'github-api',
                    'file_id': file_id,
                    'line_number': line_num,
                    'line_content': line_content,
                    'label': 'correct',
                    'label_rule': 'none',
                    'label_message': 'No issues found'
                })
        
        return labeled_lines
