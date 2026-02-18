import requests
import time
import json
import os
import base64
from datetime import datetime

class GithubDockerfileURLFetcher:
    def __init__(self, token, per_page=100):
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.per_page = per_page
        self.rate_limit_remaining = None
        self.rate_limit_reset = None

    def get_file_content_direct(self, repo_full_name, path):
        """
        Get file content directly using the GitHub Contents API
        """
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
        try:
            response = requests.get(url, headers=self.headers, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if data.get('type') == 'file' and data.get('content'):
                    content = base64.b64decode(data['content']).decode('utf-8')
                    return content
            elif response.status_code == 404:
                print(f"[DEBUG] File not found: {repo_full_name}/{path}")
            else:
                print(f"[DEBUG] HTTP {response.status_code} for {repo_full_name}/{path}")
            return None
        except Exception as e:
            print(f"[ERROR] Failed to get content for {repo_full_name}/{path}: {e}")
            return None

    def get_dockerfile_urls(self, page=1, year=None, max_pages=10):
        """
        Fetches Dockerfile content by first searching, then fetching content directly.
        Returns: (dockerfile_data_list, total_count)
        """
        if year:
            created_range = f"2025-01-01..{datetime.today().strftime('%Y-%m-%d')}"
            query = f"filename:Dockerfile+created:{created_range}"
        else:
            query = "filename:Dockerfile"
        
        all_dockerfiles = []
        total_count = 0

        for current_page in range(page, page + max_pages):
            url = f"https://api.github.com/search/code?q={query}&per_page={self.per_page}&page={current_page}"
            print(f"[FETCH] Page {current_page}")
            print(f"[DEBUG] Query: {query}")
            print(f"[DEBUG] URL: {url}")
            
            try:
                response = requests.get(url, headers=self.headers)
                
                print(f"[DEBUG] Response status: {response.status_code}")
                print(f"[DEBUG] Rate limit remaining: {response.headers.get('X-RateLimit-Remaining')}")
                
                if response.status_code == 403:
                    reset_time = response.headers.get('X-RateLimit-Reset')
                    if reset_time:
                        reset_datetime = datetime.fromtimestamp(int(reset_time))
                        print(f"[RATE_LIMIT] Rate limit exceeded. Reset at: {reset_datetime}")
                    else:
                        print("[RATE_LIMIT] Rate limit exceeded. Try again later.")
                    break
                
                if response.status_code == 401:
                    print("[ERROR] Authentication failed. Check your GitHub token.")
                    print("[ERROR] Make sure your token has the 'public_repo' scope.")
                    break
                
                if response.status_code != 200:
                    print(f"[WARN] GitHub API error {response.status_code} for {url}")
                    print(f"[DEBUG] Response body: {response.text[:500]}")
                    break

                data = response.json()
                items = data.get('items', [])
                
                print(f"[DEBUG] Found {len(items)} items in response")
                
                if not items:
                    print(f"[INFO] No more results on page {current_page}")
                    break
                
                if current_page == 1:
                    print(f"[DEBUG] Sample item keys: {list(items[0].keys()) if items else 'No items'}")
                    if items:
                        print(f"[DEBUG] First item repo: {items[0].get('repository', {}).get('full_name', 'NOT_FOUND')}")
                        print(f"[DEBUG] First item repo private: {items[0].get('repository', {}).get('private', 'UNKNOWN')}")
                
                dockerfiles = []
                for i, item in enumerate(items):
                    repo = item.get('repository', {})
                    if repo.get('private', True):
                        continue
                    
                    repo_full_name = repo['full_name']
                    path = item['path']
                    
                    if os.path.basename(path) != "Dockerfile":
                        print(f"[SKIP] Not a pure Dockerfile: {path}")
                        continue

                    print(f"[FETCH_CONTENT] {i+1}/{len(items)}: {repo_full_name}/{path}")
                    
                    content = self.get_file_content_direct(repo_full_name, path)
                    
                    if content:
                        dockerfiles.append({
                            'content': content,
                            'repo': repo_full_name,
                            'path': path,
                            'sha': item['sha'],
                            'size': item.get('size', 0)
                        })
                        print(f"[SUCCESS] Got content for {repo_full_name}/{path} ({len(content)} chars)")
                    else:
                        print(f"[FAIL] No content for {repo_full_name}/{path}")
                    
                    time.sleep(1)
                
                print(f"[DEBUG] Dockerfiles with content (public repos only): {len(dockerfiles)}")
                
                all_dockerfiles.extend(dockerfiles)
                total_count = data.get('total_count', 0)
                
                print(f"[SUCCESS] Found {len(dockerfiles)} Dockerfiles on page {current_page}")
                print(f"[INFO] Total available: {total_count}")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"[ERROR] Failed to fetch page {current_page}: {e}")
                break

        return all_dockerfiles, total_count

    def get_dockerfile_urls_by_year(self, year, max_pages=10):
        """
        Fetches up to max_pages * per_page Dockerfile content created in a given year.
        """
        return self.get_dockerfile_urls(page=1, year=year, max_pages=max_pages)

    def test_connection(self):
        """
        Test if the GitHub API connection is working
        """
        try:
            response = requests.get("https://api.github.com/user", headers=self.headers)
            if response.status_code == 200:
                user_data = response.json()
                print(f"[SUCCESS] Connected as: {user_data.get('login', 'Unknown')}")
                return True
            else:
                print(f"[ERROR] Connection failed: {response.status_code}")
                print(f"[ERROR] Response: {response.text}")
                return False
        except Exception as e:
            print(f"[ERROR] Connection test failed: {e}")
            return False
