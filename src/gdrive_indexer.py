import os
import sys
import io
import re
import json
import ssl
import time
import argparse
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure UTF-8 console output for Windows compatibility
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Create SSL context for Windows socket compatibility
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

from .config import DB_PATH, CONSOLIDATED_VECTORS_PATH, BASE_DIR
from .db import IndexDatabase


class GoogleDriveCloudIndexer:
    """
    Direct Google Drive REST API Cloud Indexer.
    Scans Google Drive folders recursively, streams only small metadata (.csv, .npy, .json, .srt)
    into RAM, while linking images directly to Google Drive CDN without local download.
    """

    def __init__(self, folder_id: str = "19CHP3l4nfOPRi9alJquvI_FJL9-T7xgA", api_key: str = "AIzaSyCs_2-bm7Duz_tctK9cvtUhTJm7vtIbmEE"):
        self.folder_id = folder_id
        self.api_key = api_key
        self.db = IndexDatabase(DB_PATH)

    def _api_get(self, url: str, retries: int = 5) -> Dict[str, Any]:
        """HTTP GET request with automatic retry and exponential backoff for socket stability."""
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, context=ssl_context, timeout=25) as res:
                    return json.loads(res.read().decode('utf-8'))
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(1.0 * (attempt + 1))
        return {}

    def _api_download_bytes(self, file_id: str, retries: int = 5) -> bytes:
        """Download file content into RAM with automatic retry."""
        url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={self.api_key}"
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
                with urllib.request.urlopen(req, context=ssl_context, timeout=35) as res:
                    return res.read()
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                time.sleep(1.0 * (attempt + 1))
        return b""

    def list_folder_children(self, parent_id: str) -> List[Dict[str, Any]]:
        """List files and subfolders inside a parent folder ID."""
        files = []
        page_token = None
        
        while True:
            encoded_query = urllib.parse.quote(f"'{parent_id}' in parents and trashed = false")
            url = f"https://www.googleapis.com/drive/v3/files?q={encoded_query}&key={self.api_key}&pageSize=1000&fields=nextPageToken,files(id,name,mimeType,size)"
            if page_token:
                url += f"&pageToken={page_token}"
            
            try:
                data = self._api_get(url)
                files.extend(data.get("files", []))
                page_token = data.get("nextPageToken")
                if not page_token:
                    break
            except Exception as e:
                print(f"[Warning] Failed to list folder {parent_id}: {e}", flush=True)
                break

        return files

    def build_gdrive_tree(self, parent_id: str, parent_path: str = "") -> List[Dict[str, Any]]:
        """Recursively scan Google Drive folder tree."""
        all_items = []
        children = self.list_folder_children(parent_id)

        for item in children:
            item_name = item["name"]
            item_id = item["id"]
            item_mime = item["mimeType"]
            rel_path = f"{parent_path}/{item_name}" if parent_path else item_name
            
            item["rel_path"] = rel_path

            if item_mime == "application/vnd.google-apps.folder":
                print(f"[Scan Folder] Subfolder: {rel_path}", flush=True)
                sub_items = self.build_gdrive_tree(item_id, rel_path)
                all_items.extend(sub_items)
            else:
                all_items.append(item)

        return all_items

    def process_gdrive_cloud(self):
        print(f"[Cloud Indexer] Connecting to Google Drive Folder ID: {self.folder_id}", flush=True)
        self.db.clear()

        print("[Cloud Indexer] Recursively scanning Google Drive folder tree via REST API...", flush=True)
        gdrive_files = self.build_gdrive_tree(self.folder_id)
        print(f"[Cloud Indexer] Total files discovered on Google Drive: {len(gdrive_files)}", flush=True)

        images_by_video: Dict[str, List[Dict[str, Any]]] = {}
        vectors_by_video: Dict[str, Dict[str, Any]] = {}
        csvs_by_video: Dict[str, Dict[str, Any]] = {}
        srts_by_video: Dict[str, Dict[str, Any]] = {}

        for item in gdrive_files:
            name = item["name"]
            rel_path = item["rel_path"]
            parts = rel_path.split("/")
            
            video_match = re.search(r'L\d+_V\d+', rel_path)
            video_id = video_match.group() if video_match else parts[0]

            ext = os.path.splitext(name)[1].lower()

            if ext in [".jpg", ".jpeg", ".png", ".webp"]:
                if video_id not in images_by_video:
                    images_by_video[video_id] = []
                images_by_video[video_id].append(item)

            elif ext == ".npy":
                vectors_by_video[video_id] = item

            elif ext == ".csv":
                csvs_by_video[video_id] = item

            elif ext == ".srt":
                srts_by_video[video_id] = item

        print(f"[Cloud Indexer] Identified {len(images_by_video)} distinct video batches on Google Drive.", flush=True)

        all_records = []
        consolidated_vectors = []
        current_global_idx = 0

        audit_report = {
            "total_video_batches": len(images_by_video),
            "indexed_keyframes": 0,
            "indexed_vectors": 0,
            "missing_components": {
                "missing_vector_npy": [],
                "missing_csv_timestamp": [],
                "missing_srt_asr": []
            }
        }

        for video_id, img_items in tqdm(images_by_video.items(), desc="Processing Google Drive Batches"):
            img_items.sort(key=lambda x: int(re.search(r'\d+', x["name"]).group()) if re.search(r'\d+', x["name"]) else x["name"])

            vectors = None
            vec_item = vectors_by_video.get(video_id)
            if vec_item:
                try:
                    vec_bytes = self._api_download_bytes(vec_item["id"])
                    vectors = np.load(io.BytesIO(vec_bytes))
                    if vectors.ndim == 1:
                        vectors = vectors.reshape(1, -1)
                except Exception as e:
                    print(f"[Warning] Failed to fetch vector {vec_item['name']}: {e}", flush=True)
            else:
                audit_report["missing_components"]["missing_vector_npy"].append(video_id)

            timestamp_map = {}
            csv_item = csvs_by_video.get(video_id)
            if csv_item:
                try:
                    csv_bytes = self._api_download_bytes(csv_item["id"])
                    df = pd.read_csv(io.BytesIO(csv_bytes))
                    for _, row in df.iterrows():
                        n_val = str(int(row.get("n", 0))).zfill(3) if "n" in row else str(row.get("frame_idx", ""))
                        pts = float(row.get("pts_time", 0.0))
                        timestamp_map[n_val] = {
                            "pts_time": pts,
                            "frame_idx": int(row.get("frame_idx", 0))
                        }
                except Exception as e:
                    print(f"[Warning] Failed to fetch CSV {csv_item['name']}: {e}", flush=True)
            else:
                audit_report["missing_components"]["missing_csv_timestamp"].append(video_id)

            for local_idx, img_item in enumerate(img_items):
                img_name = img_item["name"]
                img_id = img_item["id"]
                img_stem = Path(img_name).stem

                # Direct Google Drive CDN link
                gdrive_cdn_url = f"https://lh3.googleusercontent.com/d/{img_id}"

                ts_info = timestamp_map.get(img_stem) or timestamp_map.get(str(local_idx + 1).zfill(3)) or {}
                pts_time = ts_info.get("pts_time", round(local_idx * 3.0, 2))
                frame_idx = ts_info.get("frame_idx", local_idx + 1)

                minutes = int(pts_time // 60)
                seconds = int(pts_time % 60)
                timestamp_str = f"{minutes:02d}:{seconds:02d} ({pts_time:.1f}s)"

                vec_idx = None
                vec_dim = None
                if vectors is not None and local_idx < len(vectors):
                    vec_idx = current_global_idx
                    vec_dim = int(vectors.shape[1])
                    consolidated_vectors.append(vectors[local_idx])
                    current_global_idx += 1

                record = {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "pts_time": pts_time,
                    "timestamp": timestamp_str,
                    "image_path": gdrive_cdn_url,
                    "vector_file": vec_item["name"] if vec_item else None,
                    "vector_idx": vec_idx,
                    "vector_dim": vec_dim,
                    "ocr_text": "",
                    "asr_text": "",
                    "objects": ""
                }
                all_records.append(record)

        print(f"[Save] Saving {len(all_records)} records into local SQLite database ({DB_PATH})...", flush=True)
        self.db.insert_batch(all_records)

        if consolidated_vectors:
            vec_matrix = np.array(consolidated_vectors, dtype=np.float32)
            norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vec_matrix = vec_matrix / norms

            print(f"[Save] Saving consolidated vector matrix {vec_matrix.shape} to: {CONSOLIDATED_VECTORS_PATH}...", flush=True)
            np.save(CONSOLIDATED_VECTORS_PATH, vec_matrix)

        audit_report["indexed_keyframes"] = self.db.get_all_count()
        audit_report["indexed_vectors"] = len(consolidated_vectors)

        audit_json_path = BASE_DIR / "data_audit_report.json"
        audit_json_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")

        summary_lines = [
            "=======================================================",
            "      GOOGLE DRIVE CLOUD INDEXING & AUDIT REPORT",
            "=======================================================",
            f"Folder ID: {self.folder_id}",
            f"Total Videos Processed: {audit_report['total_video_batches']}",
            f"Total Keyframes Indexed: {audit_report['indexed_keyframes']}",
            f"Total CLIP Vectors Loaded: {audit_report['indexed_vectors']}",
            f"Missing Vector Files (.npy): {len(audit_report['missing_components']['missing_vector_npy'])}",
            f"Missing CSV Timestamps:      {len(audit_report['missing_components']['missing_csv_timestamp'])}",
            "======================================================="
        ]
        (BASE_DIR / "data_audit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        print("\n=======================================================", flush=True)
        print("      🎉 GOOGLE DRIVE CLOUD INDEXING COMPLETED", flush=True)
        print("=======================================================", flush=True)
        print(f"  📊 Total Keyframes Indexed: {self.db.get_all_count()}", flush=True)
        print(f"  ⚡ Total Vectors Loaded:    {len(consolidated_vectors)}", flush=True)
        print(f"  🌐 Image Links: Direct Google Drive CDN (0 MB local images)", flush=True)
        print(f"  📄 Audit Summary saved:     data_audit_summary.txt", flush=True)
        print("=======================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Drive REST API Direct Cloud Indexer")
    parser.add_argument("--folder_id", type=str, default="19CHP3l4nfOPRi9alJquvI_FJL9-T7xgA", help="Google Drive Folder ID")
    parser.add_argument("--api_key", type=str, default="AIzaSyCs_2-bm7Duz_tctK9cvtUhTJm7vtIbmEE", help="Google Cloud API Key")
    args = parser.parse_args()

    indexer = GoogleDriveCloudIndexer(folder_id=args.folder_id, api_key=args.api_key)
    indexer.process_gdrive_cloud()
