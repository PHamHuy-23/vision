import json
import sqlite3
import os
import sys
from pathlib import Path

def build_db():
    print("Starting migration from JSON to SQLite...")
    json_path = "G:/Desktop/web_research/frame_map_supabase.json"
    db_path = "G:/Desktop/web_research/video_index_v2.db"
    
    if os.path.exists(db_path):
        os.remove(db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create a table that stores the essential indexed fields PLUS the raw JSON for full data retrieval
    cursor.execute("""
        CREATE TABLE keyframes (
            vector_id INTEGER PRIMARY KEY,
            video_id TEXT,
            frame_idx INTEGER,
            pts_time REAL,
            asr_text TEXT,
            ocr_text TEXT,
            objects TEXT,
            raw_json TEXT
        )
    """)
    cursor.execute("CREATE INDEX idx_video_id ON keyframes(video_id)")
    cursor.execute("CREATE INDEX idx_pts_time ON keyframes(video_id, pts_time)")
    
    # Optional full text search virtual table could be added here, but for now standard LIKE is enough
    
    print(f"Loading {json_path}. This will take ~3GB of RAM and a few seconds...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except MemoryError:
        print("MemoryError! Your machine doesn't have enough RAM to load the JSON at once.")
        sys.exit(1)
        
    records = []
    print("Parsing records...")
    for fid, entry in data.items():
        v_id = entry.get("clip", {}).get("vector_id")
        if v_id is None:
            continue
            
        video_id = entry.get("video_id", "")
        
        ts_data = entry.get("timestamp") if isinstance(entry.get("timestamp"), dict) else {}
        frame_idx = ts_data.get("frame_idx", entry.get("frame_number", 0))
        pts_time = float(ts_data.get("pts_time", 0.0))
        
        asr_text = entry.get("asr", {}).get("text", "")
        ocr_text = entry.get("ocr", {}).get("text", "")
        objects = entry.get("object", {}).get("text", "")
        
        raw_json_str = json.dumps(entry, ensure_ascii=False)
        
        records.append((
            int(v_id),
            video_id,
            int(frame_idx),
            pts_time,
            asr_text,
            ocr_text,
            objects,
            raw_json_str
        ))
        
        if len(records) >= 10000:
            cursor.executemany("""
                INSERT INTO keyframes (vector_id, video_id, frame_idx, pts_time, asr_text, ocr_text, objects, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            conn.commit()
            records.clear()
            
    if records:
        cursor.executemany("""
            INSERT INTO keyframes (vector_id, video_id, frame_idx, pts_time, asr_text, ocr_text, objects, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        
    cursor.execute("SELECT COUNT(*) FROM keyframes")
    count = cursor.fetchone()[0]
    conn.close()
    
    # Free memory
    del data
    print(f"Successfully migrated {count} records to SQLite ({db_path}).")

if __name__ == "__main__":
    build_db()
