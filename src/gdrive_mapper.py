import os
import sys
import json
import urllib.request
import urllib.parse
import ssl
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

BASE_DIR = Path(__file__).resolve().parent.parent

class GDrivePathMapper:
    """
    Fast Parallel Google Drive Scanner.
    Scans Google Drive using thread pools and maps local paths (/content/drive/MyDrive/Downloaded_Files_from_Excel/...)
    to Google Drive File/Folder IDs and Web Content CDN URLs.
    """

    def __init__(self, root_folder_id: str = "19CHP3l4nfOPRi9alJquvI_FJL9-T7xgA", api_key: str = "AIzaSyCs_2-bm7Duz_tctK9cvtUhTJm7vtIbmEE", max_workers: int = 30):
        self.root_folder_id = root_folder_id
        self.api_key = api_key
        self.max_workers = max_workers
        self.path_map: Dict[str, Dict[str, str]] = {}  # rel_path -> { "id": ..., "mimeType": ..., "url": ... }
        self.prefix_to_strip = "/content/drive/MyDrive/Downloaded_Files_from_Excel/"

    def _get_children(self, parent_id: str) -> List[Dict[str, Any]]:
        files = []
        page_token = None
        query = f"'{parent_id}' in parents and trashed = false"
        encoded_query = urllib.parse.quote(query)

        while True:
            url = f"https://www.googleapis.com/drive/v3/files?q={encoded_query}&key={self.api_key}&pageSize=1000&fields=nextPageToken,files(id,name,mimeType)"
            if page_token:
                url += f"&pageToken={page_token}"

            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, context=ssl_context, timeout=25) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        files.extend(data.get('files', []))
                        page_token = data.get('nextPageToken')
                        break
                except Exception:
                    time.sleep(0.5)
            if not page_token:
                break
        return files

    def _scan_folder_node(self, folder_id: str, current_path: str):
        """Scans a single folder node and returns subfolders to scan."""
        children = self._get_children(folder_id)
        subfolders = []
        node_results = []

        for item in children:
            name = item["name"]
            item_id = item["id"]
            mime = item["mimeType"]
            rel_path = f"{current_path}/{name}".lstrip("/")

            url = f"https://lh3.googleusercontent.com/d/{item_id}" if mime != "application/vnd.google-apps.folder" else f"https://drive.google.com/drive/folders/{item_id}"

            entry = (rel_path, {
                "id": item_id,
                "mimeType": mime,
                "url": url
            })
            node_results.append(entry)

            if mime == "application/vnd.google-apps.folder":
                subfolders.append((item_id, rel_path))

        return node_results, subfolders

    def scan_drive_parallel(self):
        """Multi-threaded BFS Google Drive scanner."""
        print(f"[Scanner] Starting multi-threaded Google Drive scan (workers={self.max_workers})...", flush=True)
        t0 = time.time()
        
        queue = [(self.root_folder_id, "")]
        scanned_folders = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while queue:
                futures = {executor.submit(self._scan_folder_node, fid, path): (fid, path) for fid, path in queue}
                queue = []

                for future in as_completed(futures):
                    try:
                        node_results, subfolders = future.result()
                        scanned_folders += 1
                        for rel_path, data in node_results:
                            self.path_map[rel_path] = data
                        queue.extend(subfolders)
                    except Exception as e:
                        pass

                print(f"[Scanner] Scanned {scanned_folders} folders, {len(self.path_map)} items mapped...", flush=True)

        t1 = time.time()
        print(f"🎉 Scan finished in {t1-t0:.2f} seconds! Total items mapped: {len(self.path_map)}", flush=True)

    def convert_path(self, local_path: Optional[str]) -> Optional[Dict[str, str]]:
        if not local_path:
            return None

        clean_path = local_path.replace("\\", "/")
        if clean_path.startswith(self.prefix_to_strip):
            rel_path = clean_path[len(self.prefix_to_strip):]
        else:
            rel_path = clean_path.lstrip("/")

        match = self.path_map.get(rel_path)
        if match:
            return {
                "file_id": match["id"],
                "url": match["url"],
                "rel_path": rel_path
            }
        return {
            "file_id": None,
            "url": None,
            "rel_path": rel_path
        }

    def process_dataset_map(self, input_file: Path, output_file: Path):
        print(f"\n[Process] Converting dataset_map.json ({input_file})...", flush=True)
        with open(input_file, "r", encoding="utf-8") as f:
            ds_data = json.load(f)

        converted = {}
        for vid, entry in ds_data.items():
            converted[vid] = {
                "video_id": entry.get("video_id", vid),
                "clip": self.convert_path(entry.get("clip", {}).get("path")),
                "keyframes": {
                    "root": self.convert_path(entry.get("keyframes", {}).get("root")),
                    "count": entry.get("keyframes", {}).get("count", 0)
                },
                "objects": {
                    "root": self.convert_path(entry.get("objects", {}).get("root"))
                },
                "ocr": {
                    "root": self.convert_path(entry.get("ocr", {}).get("root"))
                },
                "timestamp": self.convert_path(entry.get("timestamp", {}).get("path")),
                "media": self.convert_path(entry.get("media", {}).get("path")),
                "subtitle": self.convert_path(entry.get("subtitle", {}).get("path")),
                "asr": {
                    "srt": self.convert_path(entry.get("asr", {}).get("srt")),
                    "txt": self.convert_path(entry.get("asr", {}).get("txt")),
                    "json": self.convert_path(entry.get("asr", {}).get("json"))
                }
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved dataset_map_supabase.json to {output_file}", flush=True)

    def process_frame_map(self, input_file: Path, output_file: Path):
        print(f"\n[Process] Converting frame_map.json ({input_file})...", flush=True)
        with open(input_file, "r", encoding="utf-8") as f:
            frame_data = json.load(f)

        converted = {}
        for fid, entry in frame_data.items():
            converted[fid] = {
                "frame_id": fid,
                "video_id": entry.get("video_id"),
                "frame_number": entry.get("frame_number"),
                "image": self.convert_path(entry.get("image")),
                "clip": {
                    "file": self.convert_path(entry.get("clip", {}).get("file")),
                    "vector_id": entry.get("clip", {}).get("vector_id")
                },
                "timestamp": entry.get("timestamp", {}),
                "object": {
                    "json": self.convert_path(entry.get("object", {}).get("json"))
                },
                "ocr": {
                    "json": self.convert_path(entry.get("ocr", {}).get("json")),
                    "txt": self.convert_path(entry.get("ocr", {}).get("txt"))
                }
            }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved frame_map_supabase.json to {output_file}", flush=True)


if __name__ == "__main__":
    mapper = GDrivePathMapper(max_workers=30)
    print("==========================================================", flush=True)
    print("   STEP 1: Fast Parallel Google Drive Path Scan & Mapping  ", flush=True)
    print("==========================================================", flush=True)
    mapper.scan_drive_parallel()

    print("\n==========================================================", flush=True)
    print("   STEP 2: Generating Supabase Mapped JSON Files           ", flush=True)
    print("==========================================================", flush=True)
    mapper.process_dataset_map(BASE_DIR / "dataset_map.json", BASE_DIR / "dataset_map_supabase.json")

    frame_map_path = BASE_DIR / "frame_map (1).json" if (BASE_DIR / "frame_map (1).json").exists() else BASE_DIR / "frame_map.json"
    mapper.process_frame_map(frame_map_path, BASE_DIR / "frame_map_supabase.json")
    print("\n🎉 STEP 1 COMPLETED SUCCESSFULLY!", flush=True)

