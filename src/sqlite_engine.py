import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import open_clip
import sqlite3
import json
import re

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .config import DATA_ROOT, CONSOLIDATED_VECTORS_PATH, CLIP_MODEL_NAME, CLIP_PRETRAINED

class SQLiteSearchEngine:
    def __init__(self, data_root: Path = DATA_ROOT, vectors_path: Path = CONSOLIDATED_VECTORS_PATH):
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

        self.data_root = Path(data_root).resolve()
        self.vectors_path = Path(vectors_path).resolve()
        self.db_path = str(self.data_root.parent / "video_index_v2.db")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.traced_text_encoder = None
        self.vectors = None
        self.faiss_index = None
        self.metadata_cache = None
        
        self._load_vectors()
        self._load_metadata_cache()
        self._init_faiss()
        self.load_clip_model()

    def _load_metadata_cache(self):
        cache_path = self.data_root.parent / "metadata_cache.pkl"
        if cache_path.exists():
            try:
                import pickle
                import time
                t0 = time.time()
                with open(cache_path, "rb") as f:
                    self.metadata_cache = pickle.load(f)
                print(f"[SQLiteEngine] In-Memory Metadata Cache loaded ({len(self.metadata_cache)} items) in {time.time() - t0:.2f}s!", flush=True)
            except Exception as e:
                print(f"[SQLiteEngine] Warning: Could not load metadata_cache.pkl: {e}", flush=True)
                self.metadata_cache = None

    def _get_metadata_item(self, v_id: int) -> Optional[Dict[str, Any]]:
        if self.metadata_cache is None:
            return None
        if isinstance(self.metadata_cache, list):
            if 0 <= v_id < len(self.metadata_cache):
                return self.metadata_cache[v_id]
        elif isinstance(self.metadata_cache, dict):
            return self.metadata_cache.get(v_id)
        return None
        
    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_vectors(self):
        if self.vectors_path.exists():
            try:
                import time
                t_vec = time.time()
                # all_vectors.npy is already L2-normalized float32 matrix
                self.vectors = np.load(self.vectors_path)
                print(f"[SQLiteSearchEngine] Loaded vector matrix: {self.vectors.shape} in {time.time() - t_vec:.2f}s", flush=True)
                
            except Exception as e:
                print(f"⚠️ Error loading vectors file {self.vectors_path}: {e}", flush=True)
                self.vectors = None

    def _init_faiss(self):
        # FAISS is extremely slow on this Windows build (10.7s), falling back to optimized NumPy (0.01s)
        self.faiss_index = None

    def load_clip_model(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        if self.model is None:
            clean_name = model_name.replace("/", "-")
            try:
                import time
                t_clip = time.time()
                print(f"[OpenCLIP] Loading text/image encoder ({clean_name})...", flush=True)
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    clean_name, pretrained=pretrained, device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(clean_name)
                self.model.eval()
                print(f"[OpenCLIP] Model loaded successfully in {time.time() - t_clip:.2f}s!", flush=True)

                # Warmup inference
                with torch.inference_mode():
                    dummy_tokens = self.tokenizer(["warmup query"]).to(self.device)
                    self.model.encode_text(dummy_tokens)
                print("[OpenCLIP] Text encoder warmed up & ready for ultra-fast CPU inference!", flush=True)
            except Exception as e:
                print(f"❌ Failed to load CLIP model: {e}", flush=True)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        if self.model is None:
            self.load_clip_model()
        tokens = self.tokenizer([text]).to(self.device)
        text_features = self.model.encode_text(tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].astype(np.float32)

    @torch.no_grad()
    def encode_image(self, image_bytes: bytes) -> np.ndarray:
        if self.model is None:
            self.load_clip_model()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        image_features = self.model.encode_image(img_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].astype(np.float32)

    def search_by_image(self, image_bytes: bytes, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []
        query_vec = self.encode_image(image_bytes)
        top_candidates = min(top_k * 10 if video_id_filter else top_k, len(self.vectors))
        if self.faiss_index is not None:
            scores_matrix, indices_matrix = self.faiss_index.search(query_vec.reshape(1, -1), top_candidates)
            scores = scores_matrix[0]
            top_indices = indices_matrix[0]
        else:
            scores_all = np.dot(self.vectors, query_vec)
            # Use argpartition for O(N) top-K selection instead of O(N log N) argsort
            top_indices = np.argpartition(scores_all, -top_candidates)[-top_candidates:]
            # Sort only the top_candidates
            top_indices = top_indices[np.argsort(scores_all[top_indices])[::-1]]
            scores = scores_all[top_indices]

        results = []
        if self.metadata_cache is not None:
            for idx, score in zip(top_indices, scores):
                idx_int = int(idx)
                meta = self._get_metadata_item(idx_int)
                if meta:
                    item = meta.copy()
                    if video_id_filter and item.get("video_id") != video_id_filter:
                        continue
                    item["score"] = float(round(score * 100, 2)) if score <= 1.0 else score
                    results.append(item)
                    if len(results) >= top_k:
                        break
            return results

        top_candidates_indices = [int(idx) for idx in top_indices]
        with self._get_db() as conn:
            cur = conn.cursor()
            placeholders = ','.join(['?'] * len(top_candidates_indices))
            if video_id_filter:
                query = f"SELECT vector_id, raw_json FROM keyframes WHERE vector_id IN ({placeholders}) AND video_id = ?"
                cur.execute(query, top_candidates_indices + [video_id_filter])
            else:
                query = f"SELECT vector_id, raw_json FROM keyframes WHERE vector_id IN ({placeholders})"
                cur.execute(query, top_candidates_indices)
            
            rows_by_vid = {row['vector_id']: row['raw_json'] for row in cur.fetchall()}
            
            for idx, score in zip(top_indices, scores):
                idx_int = int(idx)
                if idx_int in rows_by_vid:
                    results.append(self._format_result(rows_by_vid[idx_int], float(score)))
                if len(results) >= top_k:
                    break
        return results

    def _format_result(self, raw_json_str: str, score: float) -> Dict[str, Any]:
        rec = json.loads(raw_json_str)
        vid = rec.get("video_id")
        v_id = rec.get("clip", {}).get("vector_id")
        
        timestamp_data = rec.get("timestamp") if isinstance(rec.get("timestamp"), dict) else {}
        image_data = rec.get("image") if isinstance(rec.get("image"), dict) else {}
        ocr_data = rec.get("ocr") if isinstance(rec.get("ocr"), dict) else {}
        object_data = rec.get("object") if isinstance(rec.get("object"), dict) else {}

        pts = float(timestamp_data.get("pts_time", 0.0))
        minutes = int(pts // 60)
        seconds = int(pts % 60)

        gdrive_id = image_data.get("file_id")
        # Generate R2 URL from rel_path if it exists
        rel_path = image_data.get("rel_path")
        if rel_path:
            # Replace .jpg with .webp since user uploaded compressed webp images to R2
            if rel_path.endswith(".jpg"):
                rel_path = rel_path[:-4] + ".webp"
            r2_url = f"https://pub-63867f61a3cb4f34a8b442399021fbbd.r2.dev/{rel_path}"
        else:
            r2_url = ""
            
        img_url = image_data.get("url") or (f"https://lh3.googleusercontent.com/d/{gdrive_id}" if gdrive_id else "")

        ocr_txt_data = ocr_data.get("txt") if isinstance(ocr_data.get("txt"), dict) else {}
        ocr_json_data = ocr_data.get("json") if isinstance(ocr_data.get("json"), dict) else {}
        object_json_data = object_data.get("json") if isinstance(object_data.get("json"), dict) else {}

        return {
            "video_id": vid,
            "vector_id": int(v_id) if v_id is not None else 0,
            "frame_idx": timestamp_data.get("frame_idx", rec.get("frame_number")),
            "pts_time": pts,
            "timestamp": f"{minutes:02d}:{seconds:02d} ({pts:.1f}s)",
            "image_path": img_url,
            "r2_url": r2_url,
            "gdrive_file_id": gdrive_id,
            "score": float(round(score * 100, 2)) if score <= 1.0 else score,
            "ocr_text": ocr_data.get("text", ""),
            "ocr_file_id": ocr_txt_data.get("file_id", ""),
            "ocr_json_id": ocr_json_data.get("file_id", ""),
            "object_file_id": object_json_data.get("file_id", ""),
            "asr_text": rec.get("asr", {}).get("text", ""),
            "objects": object_data.get("text", ""),
            "object_detections": object_data.get("detections", []),
            "ocr_detections": ocr_data.get("detections", [])
        }

    def search(self, query_text: str = "", top_k: int = 20, video_id_filter: Optional[str] = None, query_vector_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        if query_vector_id is not None and 0 <= query_vector_id < len(self.vectors):
            query_vec = self.vectors[query_vector_id]
        else:
            try:
                from .fast_translator import fast_translator
                translated_text = fast_translator.translate(query_text)
            except Exception:
                translated_text = query_text
                
            import re
            clauses = [c.strip() for c in re.split(r',|;| and ', translated_text) if c.strip()]
            if len(clauses) > 1:
                vecs = [self.encode_text(c) for c in clauses]
                query_vec = np.mean(vecs, axis=0)
                query_vec /= np.linalg.norm(query_vec)
            else:
                query_vec = self.encode_text(translated_text)

        top_candidates = min(top_k * 10 if video_id_filter else top_k, len(self.vectors))
        if self.faiss_index is not None:
            scores_matrix, indices_matrix = self.faiss_index.search(query_vec.reshape(1, -1), top_candidates)
            scores = scores_matrix[0]
            top_indices = indices_matrix[0]
        else:
            scores_all = np.dot(self.vectors, query_vec)
            # Use argpartition for O(N) top-K selection instead of O(N log N) argsort
            top_indices = np.argpartition(scores_all, -top_candidates)[-top_candidates:]
            # Sort only the top_candidates
            top_indices = top_indices[np.argsort(scores_all[top_indices])[::-1]]
            scores = scores_all[top_indices]

        results = []
        if self.metadata_cache is not None:
            for idx, score in zip(top_indices, scores):
                idx_int = int(idx)
                meta = self._get_metadata_item(idx_int)
                if meta:
                    item = meta.copy()
                    if video_id_filter and item.get("video_id") != video_id_filter:
                        continue
                    item["score"] = float(round(score * 100, 2)) if score <= 1.0 else score
                    results.append(item)
                    if len(results) >= top_k:
                        break
            return results

        top_candidates_indices = [int(idx) for idx in top_indices]
        with self._get_db() as conn:
            cur = conn.cursor()
            placeholders = ','.join(['?'] * len(top_candidates_indices))
            if video_id_filter:
                query = f"SELECT vector_id, raw_json FROM keyframes WHERE vector_id IN ({placeholders}) AND video_id = ?"
                cur.execute(query, top_candidates_indices + [video_id_filter])
            else:
                query = f"SELECT vector_id, raw_json FROM keyframes WHERE vector_id IN ({placeholders})"
                cur.execute(query, top_candidates_indices)
            
            rows_by_vid = {row['vector_id']: row['raw_json'] for row in cur.fetchall()}
            
            for idx, score in zip(top_indices, scores):
                idx_int = int(idx)
                if idx_int in rows_by_vid:
                    results.append(self._format_result(rows_by_vid[idx_int], float(score)))
                if len(results) >= top_k:
                    break
        return results

    def search_context(self, video_id: str, frame_idx: int, limit: int = 20, surrounding: bool = False):
        results = []
        with self._get_db() as conn:
            cur = conn.cursor()
            if surrounding:
                # Query frames ordered by absolute difference
                cur.execute("""
                    SELECT raw_json FROM (
                        SELECT raw_json, frame_idx FROM keyframes 
                        WHERE video_id = ? 
                        ORDER BY ABS(frame_idx - ?) ASC LIMIT ?
                    ) ORDER BY frame_idx ASC
                """, (video_id, frame_idx, limit))
            else:
                cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx >= ? ORDER BY frame_idx ASC LIMIT ?", (video_id, frame_idx, limit))
            for row in cur.fetchall():
                results.append(self._format_result(row['raw_json'], 1.0))
        return results
    def search_frame_range(self, video_id: str, start_frame: int, end_frame: int, limit: int = 80):
        results = []
        with self._get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx >= ? AND frame_idx <= ? ORDER BY frame_idx ASC LIMIT ?", (video_id, start_frame, end_frame, limit))
            for row in cur.fetchall():
                results.append(self._format_result(row['raw_json'], 1.0))
        return results

    def search_frame_page(self, video_id: str, anchor_frame: int, direction: str = "around", limit: int = 40):
        """Page through stored keyframes by database order, not numeric frame distance."""
        limit = max(1, min(int(limit), 200))
        with self._get_db() as conn:
            cur = conn.cursor()
            if direction == "before":
                cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx < ? ORDER BY frame_idx DESC LIMIT ?", (video_id, anchor_frame, limit))
                rows = list(reversed(cur.fetchall()))
            elif direction == "after":
                cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx > ? ORDER BY frame_idx ASC LIMIT ?", (video_id, anchor_frame, limit))
                rows = cur.fetchall()
            else:
                before_count = limit // 2
                after_count = limit - before_count - 1
                cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx <= ? ORDER BY frame_idx DESC LIMIT ?", (video_id, anchor_frame, before_count + 1))
                before = list(reversed(cur.fetchall()))
                cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx > ? ORDER BY frame_idx ASC LIMIT ?", (video_id, anchor_frame, after_count))
                rows = before + cur.fetchall()
        return [self._format_result(row['raw_json'], 1.0) for row in rows]

    def search_interval(self, video_id: str, start_time: float, end_time: float, limit: int = 200):
        results = []
        with self._get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT raw_json FROM keyframes WHERE video_id = ? AND pts_time >= ? AND pts_time <= ? ORDER BY pts_time ASC LIMIT ?", (video_id, start_time, end_time, limit))
            for row in cur.fetchall():
                results.append(self._format_result(row['raw_json'], 1.0))
        return results

    def _fuzzy_text_search(self, query_text: str, field_name: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        from collections import defaultdict
        
        raw_terms = [t for t in re.findall(r'\b\w+\b', query_text.lower()) if len(t) >= 2]
        if not raw_terms:
            raw_terms = [query_text.lower()]
            
        vid_scores = defaultdict(float)
        vid_jsons = {}
        
        with self._get_db() as conn:
            cur = conn.cursor()
            for term in raw_terms:
                pattern = f"%{term}%"
                if video_id_filter:
                    cur.execute(f"SELECT vector_id, raw_json FROM keyframes WHERE video_id = ? AND {field_name} LIKE ?", (video_id_filter, pattern))
                else:
                    cur.execute(f"SELECT vector_id, raw_json FROM keyframes WHERE {field_name} LIKE ?", (pattern,))
                
                for row in cur.fetchall():
                    v_id = row['vector_id']
                    vid_scores[v_id] += 1.0
                    if v_id not in vid_jsons:
                        vid_jsons[v_id] = row['raw_json']
                        
        if not vid_scores:
            return []
            
        sorted_vids = sorted(vid_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for v_id, score in sorted_vids:
            results.append(self._format_result(vid_jsons[v_id], (score / len(raw_terms))))
            if len(results) >= top_k:
                break
        return results

    def _fts_text_search(self, query_text: str, table_name: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        clean_q = query_text.strip().replace('"', '""')
        if not clean_q:
            return []
            
        match_expr = f'"{clean_q}"'
        results = []
        
        with self._get_db() as conn:
            cur = conn.cursor()
            try:
                query = f"SELECT vector_id FROM {table_name} WHERE {table_name} MATCH ? LIMIT ?"
                cur.execute(query, (match_expr, top_k * 5 if video_id_filter else top_k))
                rows = cur.fetchall()
                
                for row in rows:
                    v_id = row['vector_id']
                    meta = self._get_metadata_item(v_id)
                    if meta:
                        item = meta.copy()
                        if video_id_filter and item.get("video_id") != video_id_filter:
                            continue
                        item["score"] = 100.0
                        results.append(item)
                    else:
                        cur2 = conn.cursor()
                        cur2.execute("SELECT raw_json FROM keyframes WHERE vector_id = ?", (v_id,))
                        r = cur2.fetchone()
                        if r:
                            results.append(self._format_result(r['raw_json'], 1.0))
                    if len(results) >= top_k:
                        break
                return results
            except Exception as e:
                print(f"[FTS5] Notice: {e}, falling back to fuzzy scan", flush=True)
                field_name = "ocr_text" if "ocr" in table_name else "asr_text"
                return self._fuzzy_text_search(query_text, field_name, top_k, video_id_filter)

    def exact_asr_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._fts_text_search(query_text, "asr_fts", top_k, video_id_filter)

    def exact_ocr_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._fts_text_search(query_text, "ocr_fts", top_k, video_id_filter)
        
    def smart_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None, enable_rerank: bool = False) -> List[Dict[str, Any]]:
        return self.search(query_text=query_text, top_k=top_k, video_id_filter=video_id_filter)
