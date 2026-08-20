import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

class IndexDatabase:
    """SQLite Database manager for Multi-Modal Keyframe Metadata indexing and retrieval."""
    
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS keyframes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id TEXT NOT NULL,
                    frame_idx INTEGER,
                    pts_time REAL,
                    timestamp TEXT,
                    image_path TEXT NOT NULL,
                    vector_file TEXT,
                    vector_idx INTEGER,
                    vector_dim INTEGER,
                    ocr_text TEXT,
                    asr_text TEXT,
                    objects TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_video_id ON keyframes(video_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frame_idx ON keyframes(frame_idx)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_vector_idx ON keyframes(vector_idx)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pts_time ON keyframes(pts_time)")
            conn.commit()

    def clear(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM keyframes")
            conn.commit()

    def insert_batch(self, records: List[Dict[str, Any]]):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany("""
                INSERT INTO keyframes (video_id, frame_idx, pts_time, timestamp, image_path, vector_file, vector_idx, vector_dim, ocr_text, asr_text, objects)
                VALUES (:video_id, :frame_idx, :pts_time, :timestamp, :image_path, :vector_file, :vector_idx, :vector_dim, :ocr_text, :asr_text, :objects)
            """, records)
            conn.commit()

    def get_all_count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM keyframes")
            row = cursor.fetchone()
            return row[0] if row else 0

    def get_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM keyframes WHERE id = ?", (item_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_by_vector_idx(self, vec_idx: int) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM keyframes WHERE vector_idx = ?", (vec_idx,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def search_text_fields(self, keyword: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Search OCR, ASR subtitle text, or detected objects using SQL LIKE query."""
        pattern = f"%{keyword}%"
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM keyframes 
                WHERE ocr_text LIKE ? OR asr_text LIKE ? OR objects LIKE ?
                LIMIT ?
            """, (pattern, pattern, pattern, limit))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
