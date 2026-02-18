import os
import requests
import hashlib
import time
import json
import base64
from urllib.parse import urlparse

class DockerfileFetcher:
    def __init__(self, output_dir="dockerfiles/github_api"):
        self.output_dir = output_dir
        self.fetched_count = 0
        self.failed_count = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_unique_id(self, repo_name, file_path):
        """
        Generate a unique ID for the Dockerfile based on repo and path
        """
        unique_string = f"{repo_name}_{file_path}"
        hash_object = hashlib.md5(unique_string.encode())
        return hash_object.hexdigest()[:8]

    def save_dockerfile_content(self, dockerfile_data):
        """
        Save Dockerfile content directly to file
        Args:
            dockerfile_data: dict with 'content', 'repo', 'path', 'sha' keys
        Returns:
            file_path if successful, None if failed
        """
        repo_name = dockerfile_data['repo']
        file_path = dockerfile_data['path']
        content = dockerfile_data['content']
        
        try:
            unique_id = self.generate_unique_id(repo_name, file_path)
            filename = f"Dockerfile_{unique_id}_{repo_name.replace('/', '_')}.docker"
            file_path_local = os.path.join(self.output_dir, filename)
            
            if os.path.exists(file_path_local):
                print(f"[SKIP] {filename} already exists")
                return file_path_local
            
            with open(file_path_local, "w", encoding="utf-8") as f:
                f.write(content)
            
            self.fetched_count += 1
            print(f"[SAVED] {filename} ({self.fetched_count} total) - Size: {len(content)} chars")
            return file_path_local
                
        except Exception as e:
            self.failed_count += 1
            print(f"[ERROR] Failed to save {repo_name}/{file_path}: {e}")
            return None

    def get_stats(self):
        """
        Return fetching statistics
        """
        return {
            'fetched': self.fetched_count,
            'failed': self.failed_count,
            'total': self.fetched_count + self.failed_count
        }

    def save_metadata(self, dockerfile_data_list, filename="fetched_metadata.json"):
        """
        Save metadata about fetched Dockerfiles
        """
        metadata = []
        for data in dockerfile_data_list:
            unique_id = self.generate_unique_id(data['repo'], data['path'])
            metadata.append({
                'unique_id': unique_id,
                'repo': data['repo'],
                'path': data['path'],
                'sha': data['sha'],
                'size': data.get('size', 0),
                'content_length': len(data['content']),
                'filename': f"Dockerfile_{unique_id}_{data['repo'].replace('/', '_')}.docker"
            })
        
        metadata_path = os.path.join(self.output_dir, filename)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[METADATA] Saved to {metadata_path}")
