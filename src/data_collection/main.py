import os
import json
import time
from src.data_collection.github_dockerfile_fetcher import GithubDockerfileURLFetcher
from src.data_collection.dockerfile_fetcher import DockerfileFetcher

def main():
    GITHUB_TOKEN = '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'
    OUTPUT_DIR = "dockerfiles"
    MAX_PAGES = 5
    YEAR = 2020

    os.environ['GITHUB_TOKEN'] = GITHUB_TOKEN

    github_fetcher = GithubDockerfileURLFetcher(GITHUB_TOKEN)
    fetcher = DockerfileFetcher(OUTPUT_DIR)

    print("[TEST] Testing GitHub API connection...")
    if not github_fetcher.test_connection():
        print("[ERROR] GitHub API connection failed. Please check your token.")
        return

    print(f"[START] Fetching Dockerfiles from GitHub...")
    print(f"[CONFIG] Output directory: {OUTPUT_DIR}")
    print(f"[CONFIG] Max pages: {MAX_PAGES}")
    if YEAR:
        print(f"[CONFIG] Year filter: {YEAR}")

    if YEAR:
        dockerfile_data_list, total_count = github_fetcher.get_dockerfile_urls_by_year(YEAR, MAX_PAGES)
    else:
        dockerfile_data_list, total_count = github_fetcher.get_dockerfile_urls(page=1, max_pages=MAX_PAGES)

    print(f"[INFO] Found {len(dockerfile_data_list)} Dockerfiles (Total available: {total_count})")

    if not dockerfile_data_list:
        print("[WARNING] No Dockerfiles found. This might be due to:")
        print("  1. Invalid GitHub token")
        print("  2. Rate limiting")
        print("  3. Search query issues")
        return

    successful_files = []
    for i, dockerfile_data in enumerate(dockerfile_data_list, 1):
        print(f"[PROGRESS] Processing {i}/{len(dockerfile_data_list)}")

        file_path = fetcher.save_dockerfile_content(dockerfile_data)
        if file_path:
            successful_files.append(file_path)

        time.sleep(0.5)

    fetcher.save_metadata(dockerfile_data_list)

    stats = fetcher.get_stats()
    print(f"\n[COMPLETE] Fetching completed!")
    print(f"[STATS] Successfully fetched: {stats['fetched']}")
    print(f"[STATS] Failed: {stats['failed']}")
    print(f"[STATS] Success rate: {stats['fetched']/(stats['fetched']+stats['failed'])*100:.1f}%")

if __name__ == "__main__":
    main()
