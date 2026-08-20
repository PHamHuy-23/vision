import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

import open_clip

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .config import DB_PATH, CONSOLIDATED_VECTORS_PATH, CLIP_MODEL_NAME, CLIP_PRETRAINED, DATA_ROOT
from .db import IndexDatabase
from . import gemini_agent
from .supabase_service import SupabaseService

class VectorSearchEngine:
    """
    High-Performance Multi-Modal Vector Search Engine powered by FAISS, OpenCLIP, and Supabase REST API.
    """

    def __init__(self, data_root: Path = DATA_ROOT, db_path: Path = DB_PATH, vectors_path: Path = CONSOLIDATED_VECTORS_PATH):
        self.data_root = Path(data_root).resolve()
        self.db = IndexDatabase(db_path)
        self.supabase_svc = SupabaseService()
        self.vectors_path = Path(vectors_path).resolve()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.vectors: Optional[np.ndarray] = None
        self.faiss_index = None
        self.vector_map: Dict[int, Dict[str, Any]] = {}

        self._load_vectors()
        self._load_metadata_map()
        self._init_faiss()
        self.load_clip_model()

    def _load_metadata_map(self):
        map_file = self.data_root.parent / "frame_map_supabase.json"
        if map_file.exists():
            try:
                import json
                print(f"[Search Engine] Pre-loading keyframe metadata map ({map_file.name})...", flush=True)
                with open(map_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for fid, entry in data.items():
                    v_id = entry.get("clip", {}).get("vector_id")
                    if v_id is not None and v_id not in self.vector_map:
                        self.vector_map[v_id] = entry
                print(f"[Search Engine] Indexed {len(self.vector_map):,} unique vector metadata entries.", flush=True)
            except Exception as e:
                print(f"⚠️ Warning loading frame_map_supabase.json: {e}", flush=True)

    def _load_vectors(self):
        if self.vectors_path.exists():
            try:
                self.vectors = np.load(self.vectors_path, mmap_mode=None).astype(np.float32)
                # Ensure L2 normalized vectors for Cosine Similarity (Inner Product)
                norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.vectors = self.vectors / norms
                print(f"[Search Engine] Loaded vector matrix: {self.vectors.shape}", flush=True)
            except Exception as e:
                print(f"⚠️ Error loading vectors file {self.vectors_path}: {e}", flush=True)
                self.vectors = None

    def _init_faiss(self):
        if HAS_FAISS and self.vectors is not None:
            try:
                dim = self.vectors.shape[1]
                # IndexFlatIP uses Inner Product (Cosine Similarity for normalized vectors)
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
                print(f"[OpenCLIP] Loading text encoder ({clean_name})...", flush=True)
                self.model, _, _ = open_clip.create_model_and_transforms(
                    clean_name, pretrained=pretrained, device=self.device
                )
                self.tokenizer = open_clip.get_tokenizer(clean_name)
                self.model.eval()
                print("[OpenCLIP] Text encoder loaded successfully!", flush=True)
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

    def search(self, query_text: str = "", top_k: int = 20, video_id_filter: Optional[str] = None, query_vector_id: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.vectors is None or len(self.vectors) == 0:
            return []

        if query_vector_id is not None and 0 <= query_vector_id < len(self.vectors):
            query_vec = self.vectors[query_vector_id]
        else:
            import re
            from deep_translator import GoogleTranslator
            
            # 1. Auto-Translate VI -> EN
            try:
                translated_text = GoogleTranslator(source='auto', target='en').translate(query_text)
                print(f"[Search] Translated: '{query_text}' -> '{translated_text}'", flush=True)
            except Exception as e:
                print('Translation error:', e)
                translated_text = query_text
                
            # 2. Non-AI Algorithm: Split into clauses/keywords for compositional CLIP search
            # Split by commas, semicolons, or the word 'and'
            clauses = [c.strip() for c in re.split(r',|;|and', translated_text) if c.strip()]
            
            if len(clauses) > 1:
                # Average the vectors of each clause (Compositional Retrieval)
                print(f"[Search] Splitting query into {len(clauses)} clauses for compositional CLIP...", flush=True)
                vecs = [self.encode_text(c) for c in clauses]
                import numpy as np
                query_vec = np.mean(vecs, axis=0)
                query_vec /= np.linalg.norm(query_vec)
            else:
                query_vec = self.encode_text(translated_text)

        # Always request more candidates from FAISS because ~31% of vector indices
        # don't have metadata in frame_map_supabase.json and will be skipped.
        top_candidates = min(top_k * 10, len(self.vectors))

        # 2. FAISS High-Speed Vector Search
        if self.faiss_index is not None:
            scores_matrix, indices_matrix = self.faiss_index.search(query_vec.reshape(1, -1), top_candidates)
            scores = scores_matrix[0]
            top_indices = indices_matrix[0]
        else:
            # Fallback to numpy dot product
            scores_all = np.dot(self.vectors, query_vec)
            top_indices = np.argsort(scores_all)[::-1][:top_candidates]
            scores = scores_all[top_indices]

        # 3. Retrieve metadata (fast local cache -> Supabase -> SQLite)
        vector_id_to_score = {int(idx): float(score) for idx, score in zip(top_indices, scores)}
        results = []

        if self.vector_map:
            for idx in top_indices:
                v_id = int(idx)
                rec = self.vector_map.get(v_id)
                if not rec:
                    continue
                vid = rec.get("video_id")
                if video_id_filter and vid != video_id_filter:
                    continue

                score = vector_id_to_score.get(v_id, 0.0)

                timestamp_data = rec.get("timestamp") or {}
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

                results.append({
                    "video_id": vid,
                    "vector_id": int(v_id),
                    "frame_idx": timestamp_data.get("frame_idx", rec.get("frame_number")),
                    "pts_time": pts,
                    "timestamp": f"{minutes:02d}:{seconds:02d} ({pts:.1f}s)",
                    "image_path": img_url,
                    "gdrive_file_id": gdrive_id,
                    "score": float(round(score * 100, 2)),
                    "ocr_text": ocr_data.get("text", ""),
                    "ocr_file_id": ocr_txt_data.get("file_id", ""),
                    "ocr_json_id": ocr_json_data.get("file_id", ""),
                    "object_file_id": object_json_data.get("file_id", ""),
                    "asr_text": "",
                    "objects": ""
                })
        else:
            vector_ids_list = [int(i) for i in top_indices]
            supabase_records = []
            if self.supabase_svc.is_configured:
                try:
                    supabase_records = self.supabase_svc.get_frames_by_vector_ids(vector_ids_list)
                except Exception as e:
                    print(f"⚠️ Supabase lookup fallback to SQLite: {e}", flush=True)

            if supabase_records:
                for rec in supabase_records:
                    vid = rec.get("video_id")
                    if video_id_filter and vid != video_id_filter:
                        continue

                    v_id = rec.get("vector_id")
                    score = vector_id_to_score.get(v_id, 0.0)
                    pts = float(rec.get("pts_time") or 0.0)
                    minutes = int(pts // 60)
                    seconds = int(pts % 60)
                    gdrive_id = rec.get("image_file_id")
                    img_url = rec.get("image_url") or f"https://lh3.googleusercontent.com/d/{gdrive_id}" if gdrive_id else ""

                    results.append({
                        "video_id": vid,
                        "frame_idx": rec.get("frame_idx", rec.get("frame_number")),
                        "pts_time": pts,
                        "timestamp": f"{minutes:02d}:{seconds:02d} ({pts:.1f}s)",
                        "image_path": img_url,
                        "gdrive_file_id": gdrive_id,
                        "score": float(round(score * 100, 2)),
                        "ocr_text": "",
                        "asr_text": "",
                        "objects": ""
                    })
            else:
                for idx in top_indices:
                    item = self.db.get_by_vector_idx(int(idx)) or self.db.get_by_id(int(idx) + 1)
                    if item:
                        if video_id_filter and item.get("video_id") != video_id_filter:
                            continue
                        item["score"] = float(round(vector_id_to_score[int(idx)] * 100, 2))
                        item["vector_id"] = int(idx)
                        results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    
    def search_context(self, video_id: str, frame_idx: int, limit: int = 20):
        results = []
        if not self.vector_map:
            return results
            
        candidates = []
        for v_id, rec in self.vector_map.items():
            vid = rec.get("video_id")
            if vid != video_id:
                continue
            ts_data = rec.get("timestamp") or {}
            f_idx = ts_data.get("frame_idx", rec.get("frame_number"))
            if f_idx >= frame_idx:
                candidates.append((f_idx, v_id, rec))
                
        candidates.sort(key=lambda x: x[0])
        top_candidates = candidates[:limit]
        
        for f_idx, v_id, rec in top_candidates:
            image_data = rec.get("image") if isinstance(rec.get("image"), dict) else {}
            ts_data = rec.get("timestamp") or {}
            pts = float(ts_data.get("pts_time", 0.0))
            minutes = int(pts // 60)
            seconds = int(pts % 60)
            gdrive_id = image_data.get("file_id")
            img_url = image_data.get("url") or (f"https://lh3.googleusercontent.com/d/{gdrive_id}" if gdrive_id else "")
            
            results.append({
                "video_id": video_id,
                "frame_idx": f_idx,
                "pts_time": pts,
                "timestamp": f"{minutes:02d}:{seconds:02d} ({pts:.1f}s)",
                "image_path": img_url,
                "gdrive_file_id": gdrive_id,
                "score": 100.0,
                "vector_id": v_id
            })
            
        return results

    def keyword_search(self, query_text: str, top_k: int = 20, video_id_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Perform text matching over OCR and Object labels stored in vector_map."""
        results = []
        if not self.vector_map:
            return results
            
        query_terms = query_text.lower().split()
        
        for v_id, rec in self.vector_map.items():
            vid = rec.get("video_id")
            if video_id_filter and vid != video_id_filter:
                continue
                
            ocr_text = rec.get("ocr", {}).get("text", "").lower()
            obj_text = rec.get("object", {}).get("text", "").lower()
            
            combined_text = ocr_text + " " + obj_text
            
            # Simple scoring: how many query terms are found
            score = 0
            for term in query_terms:
                if term in combined_text:
                    score += 1
            
            if score > 0:
                # Normalize score to a percentage-like value (0-100)
                final_score = float(round((score / len(query_terms)) * 100, 2))
                
                timestamp_data = rec.get("timestamp") or {}
                image_data = rec.get("image") or {}
                ocr_data = rec.get("ocr") or {}
                object_data = rec.get("object") or {}

                pts = float(timestamp_data.get("pts_time", 0.0))
                minutes = int(pts // 60)
                seconds = int(pts % 60)

                gdrive_id = image_data.get("file_id")
                img_url = image_data.get("url") or (f"https://lh3.googleusercontent.com/d/{gdrive_id}" if gdrive_id else "")

                ocr_txt_data = ocr_data.get("txt") or {}
                ocr_json_data = ocr_data.get("json") or {}
                object_json_data = object_data.get("json") or {}

                results.append({
                    "video_id": vid,
                    "vector_id": int(v_id),
                    "frame_idx": timestamp_data.get("frame_idx", rec.get("frame_number")),
                    "pts_time": pts,
                    "timestamp": f"{minutes:02d}:{seconds:02d} ({pts:.1f}s)",
                    "image_path": img_url,
                    "gdrive_file_id": gdrive_id,
                    "score": final_score,
                    "ocr_text": ocr_data.get("text", ""),
                    "ocr_file_id": ocr_txt_data.get("file_id", ""),
                    "ocr_json_id": ocr_json_data.get("file_id", ""),
                    "object_file_id": object_json_data.get("file_id", ""),
                    "asr_text": "",
                    "objects": object_data.get("text", "")
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def smart_search(self, query_text: str, top_k: int = 24, video_id_filter: Optional[str] = None, enable_rerank: bool = False) -> List[Dict[str, Any]]:
        """Hybrid search using Gemini for query expansion (with graceful fallback)."""
        import time
        t0 = time.time()
        
        # 1. Expand query via Gemini (graceful fallback: use original query for CLIP)
        expanded = gemini_agent.expand_query(query_text)
        semantic_q = expanded.get("semantic_query", query_text)
        obj_kw = expanded.get("object_keywords", [])
        ocr_kw = expanded.get("ocr_keywords", [])
        
        t1 = time.time()
        print(f"[Smart Search] Step 1 - Query Expansion: {t1-t0:.2f}s | semantic_q='{semantic_q}' | obj={obj_kw} | ocr={ocr_kw}", flush=True)
        
        # 2. Get Semantic results (this is the core — always works)
        semantic_results = self.search(semantic_q, top_k=top_k*2, video_id_filter=video_id_filter)
        
        t2 = time.time()
        print(f"[Smart Search] Step 2 - Semantic Search: {t2-t1:.2f}s | {len(semantic_results)} results", flush=True)
        
        # 3. Get Keyword results if keywords exist
        kw_results = []
        kw_string = " ".join(obj_kw + ocr_kw)
        if kw_string.strip():
            kw_results = self.keyword_search(kw_string, top_k=top_k*2, video_id_filter=video_id_filter)
            
        t3 = time.time()
        print(f"[Smart Search] Step 3 - Keyword Search: {t3-t2:.2f}s | {len(kw_results)} results", flush=True)
            
        # 4. Merge results (Hybrid scoring)
        merged_dict = {}
        for r in semantic_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            r["semantic_score"] = r["score"]
            r["keyword_score"] = 0
            r["score"] = r["semantic_score"] # initial score
            merged_dict[key] = r
            
        for r in kw_results:
            key = f"{r['video_id']}_{r['frame_idx']}"
            if key in merged_dict:
                merged_dict[key]["keyword_score"] = r["score"]
                # Boost score
                merged_dict[key]["score"] = merged_dict[key]["semantic_score"] + (merged_dict[key]["keyword_score"] * 0.5)
            else:
                r["semantic_score"] = 0
                r["keyword_score"] = r["score"]
                r["score"] = r["keyword_score"] * 0.8 # penalty for no semantic match
                merged_dict[key] = r
                
        final_results = list(merged_dict.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        t4 = time.time()
        print(f"[Smart Search] Step 4 - Merge: {t4-t3:.2f}s | {len(final_results)} merged results", flush=True)
        
        # 5. Vision Re-ranking (optional — skipped by default to avoid slow thumbnail downloads + API quota issues)
        if enable_rerank:
            top_candidates = final_results[:max(16, top_k)] 
            reranked_results = gemini_agent.rerank_images(query_text, top_candidates)
            t5 = time.time()
            print(f"[Smart Search] Step 5 - Rerank: {t5-t4:.2f}s", flush=True)
            return reranked_results[:top_k]
        
        print(f"[Smart Search] Total: {time.time()-t0:.2f}s (rerank=off)", flush=True)
        return final_results[:top_k]

    def temporal_search(self, queries: List[str], top_k: int = 24, max_frame_gap: int = 900) -> List[Dict[str, Any]]:
        """
        Searches for a sequence of events. E.g. queries = ["car stops", "man gets out"]
        We search each independently, then find pairs in the same video where query2 happens AFTER query1
        within a max_frame_gap (e.g. 900 frames = ~30 seconds).
        """
        if len(queries) < 2:
            return self.smart_search(queries[0], top_k=top_k)

        # Get top 500 for the first query to ensure we have a good pool
        # We disable re-ranking here for speed by directly calling semantic/keyword logic
        # But to keep it simple, we'll just use self.search (Semantic only) for temporal parts to save time.
        q1_results = self.search(queries[0], top_k=500)
        q2_results = self.search(queries[1], top_k=500)

        # Group Q2 results by video_id for fast lookup
        q2_by_vid = {}
        for r2 in q2_results:
            vid = r2["video_id"]
            if vid not in q2_by_vid:
                q2_by_vid[vid] = []
            q2_by_vid[vid].append(r2)

        matched_sequences = []
        
        for r1 in q1_results:
            vid = r1["video_id"]
            if vid not in q2_by_vid:
                continue
                
            frame1 = r1["frame_idx"]
            
            # Find a matching r2 that happens AFTER r1 within max_frame_gap
            for r2 in q2_by_vid[vid]:
                frame2 = r2["frame_idx"]
                if 0 < (frame2 - frame1) <= max_frame_gap:
                    # Found a valid temporal sequence!
                    combined_score = r1["score"] + r2["score"]
                    
                    # Create a new result item based on r1, but add temporal metadata
                    seq_result = r1.copy()
                    seq_result["score"] = combined_score / 2.0  # Average score
                    seq_result["temporal_match"] = f"Next event at Frame {frame2} ({r2['timestamp']})"
                    matched_sequences.append(seq_result)
                    break # Just take the first valid match for this r1
                    
        # Sort by combined score
        matched_sequences.sort(key=lambda x: x["score"], reverse=True)
        return matched_sequences[:top_k]


