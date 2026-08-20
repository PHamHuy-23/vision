import os
import sys
import json
import ssl
import time
import urllib.request
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.stdout.reconfigure(encoding='utf-8')

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

BASE_DIR = Path(__file__).resolve().parent.parent

class SupabaseUploader:
    """
    Fast Parallel Ingestion for Supabase REST API (videos & frames tables).
    """

    def __init__(self, supabase_url: str = None, supabase_key: str = None, max_workers: int = 15):
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL", "")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_KEY", "")
        self.max_workers = max_workers

        if not self.supabase_url or not self.supabase_key:
            print("[Warning] SUPABASE_URL or SUPABASE_KEY environment variable is missing.", flush=True)

    def _send_chunk(self, endpoint: str, headers: dict, chunk: List[Dict[str, Any]]) -> int:
        body_bytes = json.dumps(chunk, ensure_ascii=False).encode('utf-8')
        for attempt in range(4):
            try:
                req = urllib.request.Request(endpoint, data=body_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, context=ssl_context, timeout=30) as resp:
                    return len(chunk)
            except Exception as e:
                if attempt == 3:
                    print(f"[Error Uploading Chunk] {e}", flush=True)
                    return 0
                time.sleep(1.0 * (attempt + 1))
        return 0

    def _post_batch_parallel(self, table_name: str, records: List[Dict[str, Any]], batch_size: int = 1000):
        if not self.supabase_url or not self.supabase_key:
            print(f"[Skip] Cannot upload to '{table_name}': Missing Supabase credentials.", flush=True)
            return

        endpoint = f"{self.supabase_url.rstrip('/')}/rest/v1/{table_name}"
        headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "resolution=merge-duplicates"
        }

        total = len(records)
        chunks = [records[i:i + batch_size] for i in range(0, total, batch_size)]
        print(f"[Supabase Upload] Uploading {total} records in {len(chunks)} chunks into public.{table_name} (workers={self.max_workers})...", flush=True)

        uploaded_count = 0
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._send_chunk, endpoint, headers, chunk) for chunk in chunks]
            for i, future in enumerate(as_completed(futures)):
                count = future.result()
                uploaded_count += count
                if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                    print(f"   Progress: {uploaded_count} / {total} records uploaded ({((uploaded_count/total)*100):.1f}%)", flush=True)

        t1 = time.time()
        print(f"🎉 Upload finished for public.{table_name} in {t1-t0:.2f}s! ({uploaded_count}/{total} uploaded)\n", flush=True)

    def upload_videos(self, dataset_map_file: Path):
        if not dataset_map_file.exists():
            print(f"[Error] File not found: {dataset_map_file}", flush=True)
            return

        with open(dataset_map_file, "r", encoding="utf-8") as f:
            ds_data = json.load(f)

        records = []
        for vid, entry in ds_data.items():
            records.append({
                "video_id": vid,
                "keyframes_count": entry.get("keyframes", {}).get("count", 0),
                "clip_file_id": (entry.get("clip") or {}).get("file_id"),
                "clip_url": (entry.get("clip") or {}).get("url"),
                "keyframes_folder_id": (entry.get("keyframes", {}).get("root") or {}).get("file_id"),
                "objects_folder_id": (entry.get("objects", {}).get("root") or {}).get("file_id"),
                "ocr_folder_id": (entry.get("ocr", {}).get("root") or {}).get("file_id"),
                "timestamp_file_id": (entry.get("timestamp") or {}).get("file_id"),
                "media_file_id": (entry.get("media") or {}).get("file_id"),
                "subtitle_srt_id": (entry.get("subtitle") or {}).get("file_id"),
                "asr_json_id": (entry.get("asr", {}).get("json") or {}).get("file_id")
            })

        self._post_batch_parallel("videos", records, batch_size=200)

    def upload_frames(self, frame_map_file: Path):
        if not frame_map_file.exists():
            print(f"[Error] File not found: {frame_map_file}", flush=True)
            return

        print(f"[Loading] Reading frame_map_supabase.json...", flush=True)
        t0 = time.time()
        with open(frame_map_file, "r", encoding="utf-8") as f:
            frame_data = json.load(f)
        print(f"Read {len(frame_data)} entries in {time.time()-t0:.2f}s", flush=True)

        records = []
        for fid, entry in frame_data.items():
            records.append({
                "frame_id": fid,
                "video_id": entry.get("video_id"),
                "frame_number": entry.get("frame_number"),
                "image_file_id": (entry.get("image") if isinstance(entry.get("image"), dict) else {}).get("file_id"),
                "image_url": (entry.get("image") if isinstance(entry.get("image"), dict) else {}).get("url"),
                "vector_id": (entry.get("clip") if isinstance(entry.get("clip"), dict) else {}).get("vector_id"),
                "pts_time": entry.get("timestamp", {}).get("pts_time"),
                "fps": entry.get("timestamp", {}).get("fps"),
                "frame_idx": entry.get("timestamp", {}).get("frame_idx"),
                "object_json_id": (entry.get("object", {}).get("json") if isinstance(entry.get("object", {}).get("json"), dict) else {}).get("file_id"),
                "ocr_json_id": (entry.get("ocr", {}).get("json") if isinstance(entry.get("ocr", {}).get("json"), dict) else {}).get("file_id"),
                "ocr_txt_id": (entry.get("ocr", {}).get("txt") if isinstance(entry.get("ocr", {}).get("txt"), dict) else {}).get("file_id")
            })

        self._post_batch_parallel("frames", records, batch_size=1000)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "https://qtuqhgdwtbkwjjhfknip.supabase.co"
    key = sys.argv[2] if len(sys.argv) > 2 else "sb_publishable_tv8ibMQD2lLT4fw7ocEYIA_BGjPnY68"

    uploader = SupabaseUploader(supabase_url=url, supabase_key=key, max_workers=15)
    print("==========================================================", flush=True)
    print("   STEP 2: Fast Parallel Ingestion to Supabase REST API    ", flush=True)
    print("==========================================================", flush=True)
    uploader.upload_videos(BASE_DIR / "dataset_map_supabase.json")
    uploader.upload_frames(BASE_DIR / "frame_map_supabase.json")
    print("🎉 STEP 2 COMPLETED SUCCESSFULLY!", flush=True)

