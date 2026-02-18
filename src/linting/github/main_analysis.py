import os
from pathlib import Path
from src.linting.github.dockerfile_linter import HadolintLinter
from src.linting.github.excel_saver import ExcelSaver

def main():
    GITHUB_DIR = "dockerfiles/github_api"
    OUTPUT_FILE = "dockerfile_analysis.csv"
    MAX_FILES = 5
    
    linter = HadolintLinter()
    saver = ExcelSaver(OUTPUT_FILE)
    
    try:
        import subprocess
        result = subprocess.run(["hadolint", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("[ERROR] Hadolint not found. Please install it first:")
            print("  curl -L https://github.com/hadolint/hadolint/releases/latest/download/hadolint-Linux-x86_64 -o hadolint")
            print("  chmod +x hadolint")
            print("  sudo mv hadolint /usr/local/bin/")
            return
        print(f"[INFO] Using Hadolint: {result.stdout.strip()}")
    except Exception as e:
        print(f"[ERROR] Failed to check Hadolint: {e}")
        return
    
    github_path = Path(GITHUB_DIR)
    if not github_path.exists():
        print(f"[ERROR] Directory not found: {GITHUB_DIR}")
        return
    
    dockerfiles = list(github_path.glob("Dockerfile_*"))
    print(f"[INFO] Found {len(dockerfiles)} Dockerfiles in {GITHUB_DIR}")
    
    files_to_process = dockerfiles
    print(f"[INFO] Processing {len(files_to_process)} files for sample")
    
    for i, dockerfile_path in enumerate(files_to_process, 1):
        print(f"\n[PROGRESS] Processing {i}/{len(files_to_process)}: {dockerfile_path.name}")
        
        try:
            lint_results = linter.lint_file(dockerfile_path)
            print(f"[INFO] Found {len(lint_results)} lint issues")
            
            labeled_lines = linter.get_line_labels(dockerfile_path, lint_results)
            print(f"[INFO] Processed {len(labeled_lines)} lines")
            
            saver.add_records(labeled_lines)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {dockerfile_path.name}: {e}")
            continue
    
    print(f"\n[SAVING] Saving results to {OUTPUT_FILE}...")
    saver.save()
    
    print(f"\n[COMPLETE] Analysis finished! Check {OUTPUT_FILE} for results.")

if __name__ == "__main__":
    main()
