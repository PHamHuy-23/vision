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
        self.data_root = Path(data_root).resolve()
        self.vectors_path = Path(vectors_path).resolve()
        self.db_path = str(self.data_root.parent / "video_index_v2.db")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.vectors = None
        self.faiss_index = None
        
        self._load_vectors()
        self._init_faiss()
        self.load_clip_model()
        
    def _get_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_vectors(self):
        if self.vectors_path.exists():
            try:
                self.vectors = np.load(self.vectors_path, mmap_mode='r').astype(np.float32)
                # We can't normalize a read-only mmap in place, so we copy it
                vectors_copy = np.copy(self.vectors)
                norms = np.linalg.norm(vectors_copy, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.vectors = vectors_copy / norms
                print(f"[SQLiteSearchEngine] Loaded vector matrix: {self.vectors.shape}", flush=True)
            except Exception as e:
                print(f"⚠️ Error loading vectors file {self.vectors_path}: {e}", flush=True)
                self.vectors = None

    def _init_faiss(self):
        if HAS_FAISS and self.vectors is not None:
            try:
                dim = self.vectors.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                self.faiss_index.add(self.vectors)
                print(f"[FAISS] Initialized FAISS IndexFlatIP with {self.faiss_index.ntotal} vectors (dim={dim}).", flush=True)
            except Exception as e:
                print(f"⚠️ Failed initializing FAISS index: {e}", flush=True)
                self.faiss_index = None

    def load_clip_model(self, model_name: str = CLIP_MODEL_NAME, pretrained: str = CLIP_PRETRAINED):
        if self.model is None:
            clean_name = model_name.replace("/", "-")
            try:
                print(f"[OpenCLIP] Loading text/image encoder ({clean_name})...", flush=True)
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    clean_name, pretrained=pretrained, device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(clean_name)
                self.model.eval()
                print("[OpenCLIP] Model loaded successfully!", flush=True)
            except Exception as e:
                print(f"❌ Failed to load CLIP model: {e}", flush=True)

    @torch.no_grad()
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
        top_candidates = min(top_k * 10, len(self.vectors))
        if self.faiss_index is not None:
            scores_matrix, indices_matrix = self.faiss_index.search(query_vec.reshape(1, -1), top_candidates)
            scores = scores_matrix[0]
            top_indices = indices_matrix[0]
        else:
            scores_all = np.dot(self.vectors, query_vec)
            top_indices = np.argsort(scores_all)[::-1][:top_candidates]
            scores = scores_all[top_indices]

        results = []
        with self._get_db() as conn:
            cur = conn.cursor()
            for idx, score in zip(top_indices, scores):
                if video_id_filter:
                    cur.execute("SELECT raw_json FROM keyframes WHERE vector_id = ? AND video_id = ?", (int(idx), video_id_filter))
                else:
                    cur.execute("SELECT raw_json FROM keyframes WHERE vector_id = ?", (int(idx),))
                row = cur.fetchone()
                if row:
                    results.append(self._format_result(row['raw_json'], float(score)))
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
            from deep_translator import GoogleTranslator
            try:
                translated_text = GoogleTranslator(source='auto', target='en').translate(query_text)
            except Exception:
                translated_text = query_text
                
            clauses = [c.strip() for c in re.split(r',|;| and ', translated_text) if c.strip()]
            if len(clauses) > 1:
                vecs = [self.encode_text(c) for c in clauses]
                query_vec = np.mean(vecs, axis=0)
                query_vec /= np.linalg.norm(query_vec)
            else:
                query_vec = self.encode_text(translated_text)

        top_candidates = min(top_k * 10, len(self.vectors))
        if self.faiss_index is not None:
            scores_matrix, indices_matrix = self.faiss_index.search(query_vec.reshape(1, -1), top_candidates)
            scores = scores_matrix[0]
            top_indices = indices_matrix[0]
        else:
            scores_all = np.dot(self.vectors, query_vec)
            top_indices = np.argsort(scores_all)[::-1][:top_candidates]
            scores = scores_all[top_indices]

        results = []
        with self._get_db() as conn:
            cur = conn.cursor()
            for idx, score in zip(top_indices, scores):
                if video_id_filter:
                    cur.execute("SELECT raw_json FROM keyframes WHERE vector_id = ? AND video_id = ?", (int(idx), video_id_filter))
                else:
                    cur.execute("SELECT raw_json FROM keyframes WHERE vector_id = ?", (int(idx),))
                row = cur.fetchone()
                if row:
                    results.append(self._format_result(row['raw_json'], float(score)))
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

    def exact_asr_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._fuzzy_text_search(query_text, "asr_text", top_k, video_id_filter)

    def exact_ocr_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._fuzzy_text_search(query_text, "ocr_text", top_k, video_id_filter)
        
    def smart_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None, enable_rerank: bool = False) -> List[Dict[str, Any]]:
        return self.search(query_text=query_text, top_k=top_k, video_id_filter=video_id_filter)

    def temporal_search(self, queries: List[str], top_k: int = 24, max_frame_gap: int = 900) -> List[Dict[str, Any]]:
        if len(queries) < 2:
            return self.search(queries[0], top_k=top_k)
        q1_results = self.search(queries[0], top_k=200)
        q2_results = self.search(queries[1], top_k=200)
        
        q2_by_vid = {}
        for r2 in q2_results:
            vid = r2["video_id"]
            if vid not in q2_by_vid: q2_by_vid[vid] = []
            q2_by_vid[vid].append(r2)

        matched_sequences = []
        for r1 in q1_results:
            vid = r1["video_id"]
            if vid not in q2_by_vid: continue
            frame1 = r1["frame_idx"]
            for r2 in q2_by_vid[vid]:
                frame2 = r2["frame_idx"]
                if 0 < (frame2 - frame1) <= max_frame_gap:
                    combined_score = r1["score"] + r2["score"]
                    seq_result = r1.copy()
                    seq_result["score"] = combined_score / 2.0
                    seq_result["temporal_match"] = f"Next event at Frame {frame2} ({r2['timestamp']})"
                    matched_sequences.append(seq_result)
                    break
        matched_sequences.sort(key=lambda x: x["score"], reverse=True)
        return matched_sequences[:top_k]
