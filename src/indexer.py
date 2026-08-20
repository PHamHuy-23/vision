import os
import sys
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure UTF-8 output encoding for Windows console compatibility
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from .config import (
    DATA_ROOT, KEYFRAMES_DIR_NAME, DB_PATH, CONSOLIDATED_VECTORS_PATH,
    IMAGE_EXTENSIONS, CSV_EXTENSIONS, BASE_DIR
)
from .db import IndexDatabase


def parse_srt_subtitles(srt_path: Path) -> List[Dict[str, Any]]:
    """Parse SRT subtitle file safely. Returns empty list if missing or corrupt."""
    subtitles = []
    if not srt_path.exists():
        return subtitles

    try:
        content = srt_path.read_text(encoding="utf-8", errors="ignore")
        blocks = re.split(r'\n\s*\n', content.strip())
        for block in blocks:
            lines = [l.strip() for l in block.split('\n') if l.strip()]
            if len(lines) >= 3:
                time_line = lines[1]
                text = " ".join(lines[2:])
                time_match = re.match(r'(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)', time_line)
                if time_match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
                    start_sec = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000.0
                    end_sec = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000.0
                    subtitles.append({
                        "start": start_sec,
                        "end": end_sec,
                        "text": text
                    })
    except Exception as e:
        print(f"[Warning] Error reading SRT file {srt_path.name}: {e}")

    return subtitles


def get_subtitle_text_for_pts(subtitles: List[Dict[str, Any]], pts_time: float) -> str:
    """Find ASR speech text corresponding to a keyframe timestamp."""
    matching_texts = []
    for sub in subtitles:
        if sub["start"] <= pts_time <= sub["end"] or (sub["start"] - 1.0 <= pts_time <= sub["end"] + 1.0):
            matching_texts.append(sub["text"])
    return " | ".join(matching_texts) if matching_texts else ""


def parse_objects_json(obj_json_path: Path) -> str:
    """Extract object class entities from object detection JSON safely."""
    if not obj_json_path.exists():
        return ""

    try:
        data = json.loads(obj_json_path.read_text(encoding="utf-8", errors="ignore"))
        entities = data.get("detection_class_entities", [])
        scores = data.get("detection_scores", [])
        
        valid_entities = set()
        for idx, entity in enumerate(entities):
            score = float(scores[idx]) if idx < len(scores) else 1.0
            if score >= 0.2:
                valid_entities.add(str(entity).strip())

        return ", ".join(sorted(valid_entities))
    except Exception:
        return ""


def parse_ocr_text(ocr_path: Path) -> str:
    """Read OCR text from .txt or .json file safely."""
    if not ocr_path or not ocr_path.exists():
        return ""

    try:
        if ocr_path.suffix.lower() == ".txt":
            return ocr_path.read_text(encoding="utf-8", errors="ignore").strip()
        elif ocr_path.suffix.lower() == ".json":
            data = json.loads(ocr_path.read_text(encoding="utf-8", errors="ignore"))
            return data.get("full_text", "").strip()
    except Exception:
        pass
    return ""


def process_and_index(data_root: str):
    root_path = Path(data_root).resolve()
    if not root_path.exists():
        print(f"[Error] Root directory does not exist: {root_path}")
        return

    db = IndexDatabase(DB_PATH)
    db.clear()

    print(f"[Scan] Recursive scanning of dataset root: {root_path}")

    # Recursive search for video directories
    video_dirs = [d for d in root_path.rglob("*") if d.is_dir() and ((d / f"{d.name}.csv").exists() or (d / f"{d.name}.json").exists())]
    if not video_dirs:
        video_dirs = list({img.parent for img in root_path.rglob("*.jpg")})

    print(f"[Scan] Detected {len(video_dirs)} video batches.")

    all_records = []
    consolidated_vectors = []
    current_global_idx = 0

    # Audit tracking report structure
    audit_report = {
        "total_video_batches": len(video_dirs),
        "indexed_keyframes": 0,
        "indexed_vectors": 0,
        "missing_components": {
            "missing_vector_npy": [],
            "missing_csv_timestamp": [],
            "missing_srt_asr": [],
            "missing_ocr_dir": [],
            "missing_objects_json_count": 0
        },
        "video_summary": []
    }

    for v_dir in tqdm(video_dirs, desc="Indexing Video Batches"):
        video_id = v_dir.name
        v_missing = []

        # 1. Read Vector file (.npy)
        vec_file = v_dir / f"{video_id}.npy"
        if not vec_file.exists():
            vec_file_candidates = list(v_dir.glob("*.npy"))
            if vec_file_candidates:
                vec_file = vec_file_candidates[0]

        vectors = None
        if vec_file.exists():
            try:
                vectors = np.load(vec_file)
                if vectors.ndim == 1:
                    vectors = vectors.reshape(1, -1)
            except Exception as e:
                print(f"[Warning] Error loading vector file {vec_file.name}: {e}")
                v_missing.append("Corrupt .npy vector file")
        else:
            v_missing.append("Missing .npy vector file")
            audit_report["missing_components"]["missing_vector_npy"].append(video_id)

        # 2. Read CSV mapping (timestamp)
        csv_file = v_dir / f"{video_id}.csv"
        timestamp_map = {}
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
                for _, row in df.iterrows():
                    n_val = str(int(row.get("n", 0))).zfill(3) if "n" in row else str(row.get("frame_idx", ""))
                    pts = float(row.get("pts_time", 0.0))
                    timestamp_map[n_val] = {
                        "pts_time": pts,
                        "frame_idx": int(row.get("frame_idx", 0))
                    }
            except Exception as e:
                print(f"[Warning] Error reading CSV {csv_file.name}: {e}")
                v_missing.append("Corrupt CSV timestamp")
        else:
            v_missing.append("Missing CSV timestamp (using fallback)")
            audit_report["missing_components"]["missing_csv_timestamp"].append(video_id)

        # 3. Read Subtitles (SRT / ASR)
        srt_file = v_dir / f"{video_id}.srt"
        subtitles = parse_srt_subtitles(srt_file)
        if not subtitles:
            audit_report["missing_components"]["missing_srt_asr"].append(video_id)
            v_missing.append("Missing SRT/ASR subtitle")

        # 4. Locate keyframe images (numbered 001.jpg, 002.jpg...)
        img_dir = v_dir / video_id if (v_dir / video_id).is_dir() else v_dir
        image_files = sorted(
            [p for p in img_dir.glob("*.jpg")],
            key=lambda p: int(re.search(r'\d+', p.stem).group()) if re.search(r'\d+', p.stem) else p.name
        )

        ocr_dir = v_dir / f"ocr_{video_id}"
        if not ocr_dir.exists():
            audit_report["missing_components"]["missing_ocr_dir"].append(video_id)
            v_missing.append("Missing OCR directory")

        # 5. Process each sequential image keyframe
        video_frame_count = len(image_files)
        video_missing_objects_count = 0

        for local_idx, img_path in enumerate(image_files):
            img_stem = img_path.stem
            
            # Timestamp info with fallback
            ts_info = timestamp_map.get(img_stem) or timestamp_map.get(str(local_idx + 1).zfill(3)) or {}
            pts_time = ts_info.get("pts_time", round(local_idx * 3.0, 2))
            frame_idx = ts_info.get("frame_idx", local_idx + 1)
            
            minutes = int(pts_time // 60)
            seconds = int(pts_time % 60)
            timestamp_str = f"{minutes:02d}:{seconds:02d} ({pts_time:.1f}s)"

            # OCR text (fallback to empty)
            ocr_file = ocr_dir / f"{img_stem}_ocr.txt" if ocr_dir.exists() else None
            if not ocr_file or not ocr_file.exists():
                ocr_file = ocr_dir / f"{img_stem}_ocr.json" if ocr_dir.exists() else None
            ocr_text = parse_ocr_text(ocr_file) if ocr_file else ""

            # Object entities (fallback to empty)
            obj_file = img_dir / f"{img_stem}.json"
            if not obj_file.exists():
                video_missing_objects_count += 1
            objects_text = parse_objects_json(obj_file)

            # ASR Speech text (fallback to empty)
            asr_text = get_subtitle_text_for_pts(subtitles, pts_time)

            # Vector index (fallback to None if vector missing)
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
                "image_path": str(img_path.relative_to(root_path)),
                "vector_file": str(vec_file.relative_to(root_path)) if vec_file and vec_file.exists() else None,
                "vector_idx": vec_idx,
                "vector_dim": vec_dim,
                "ocr_text": ocr_text,
                "asr_text": asr_text,
                "objects": objects_text
            }
            all_records.append(record)

        audit_report["missing_components"]["missing_objects_json_count"] += video_missing_objects_count
        audit_report["video_summary"].append({
            "video_id": video_id,
            "total_keyframes": video_frame_count,
            "has_vectors": vectors is not None,
            "missing_issues": v_missing
        })

    # Save to SQLite Database
    print(f"[Save] Saving {len(all_records)} records into local SQLite DB ({DB_PATH})...")
    db.insert_batch(all_records)

    # Save consolidated embeddings array
    if consolidated_vectors:
        vec_matrix = np.array(consolidated_vectors, dtype=np.float32)
        norms = np.linalg.norm(vec_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vec_matrix = vec_matrix / norms

        print(f"[Save] Saving consolidated embedding matrix {vec_matrix.shape} to: {CONSOLIDATED_VECTORS_PATH}...")
        np.save(CONSOLIDATED_VECTORS_PATH, vec_matrix)

    audit_report["indexed_keyframes"] = db.get_all_count()
    audit_report["indexed_vectors"] = len(consolidated_vectors)

    # Export Audit Report JSON & Human-readable Summary
    audit_json_path = BASE_DIR / "data_audit_report.json"
    audit_json_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")

    generate_audit_summary_txt(audit_report, BASE_DIR / "data_audit_summary.txt")

    print("\n=======================================================")
    print("      🎉 DATA INDEXING & AUDIT COMPLETED")
    print("=======================================================")
    print(f"  📊 Total Keyframes Indexed: {db.get_all_count()}")
    print(f"  ⚡ Total Vectors Loaded:    {len(consolidated_vectors)}")
    print(f"  ⚠️ Videos missing .npy vector: {len(audit_report['missing_components']['missing_vector_npy'])}")
    print(f"  ⚠️ Videos missing CSV map:    {len(audit_report['missing_components']['missing_csv_timestamp'])}")
    print(f"  ⚠️ Videos missing Subtitles:  {len(audit_report['missing_components']['missing_srt_asr'])}")
    print(f"  ⚠️ Videos missing OCR:        {len(audit_report['missing_components']['missing_ocr_dir'])}")
    print(f"  📄 Full Audit Report saved:   data_audit_report.json")
    print(f"  📄 Summary Report saved:      data_audit_summary.txt")
    print("=======================================================\n")


def generate_audit_summary_txt(report: Dict[str, Any], output_path: Path):
    """Generate a clear human-readable text audit report."""
    lines = [
        "=======================================================",
        "      BÁO CÁO KIỂM TOÁN THẤT THOÁT VÀ THIẾU SÓT DỮ LIỆU",
        "=======================================================",
        f"Tổng số Video batches phát hiện: {report['total_video_batches']}",
        f"Tổng số Keyframes đã nạp CSDL:  {report['indexed_keyframes']}",
        f"Tổng số Vectors đặc trưng CLIP:  {report['indexed_vectors']}",
        "",
        "--- CHI TIẾT THÀNH PHẦN THIẾU SÓT (MISSING COMPONENTS) ---",
        f"1. Số Video thiếu file Vector (.npy): {len(report['missing_components']['missing_vector_npy'])}",
        f"   Danh sách: {', '.join(report['missing_components']['missing_vector_npy']) if report['missing_components']['missing_vector_npy'] else 'Không có (Đầy đủ)'}",
        "",
        f"2. Số Video thiếu file CSV Timestamp: {len(report['missing_components']['missing_csv_timestamp'])}",
        f"   Danh sách: {', '.join(report['missing_components']['missing_csv_timestamp']) if report['missing_components']['missing_csv_timestamp'] else 'Không có (Đầy đủ)'}",
        "",
        f"3. Số Video thiếu Phụ đề ASR (.srt): {len(report['missing_components']['missing_srt_asr'])}",
        f"   Danh sách: {', '.join(report['missing_components']['missing_srt_asr']) if report['missing_components']['missing_srt_asr'] else 'Không có (Đầy đủ)'}",
        "",
        f"4. Số Video thiếu Thư mục OCR:       {len(report['missing_components']['missing_ocr_dir'])}",
        f"   Danh sách: {', '.join(report['missing_components']['missing_ocr_dir']) if report['missing_components']['missing_ocr_dir'] else 'Không có (Đầy đủ)'}",
        "",
        f"5. Số Frame thiếu Object JSON:       {report['missing_components']['missing_objects_json_count']}",
        "======================================================="
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Indexer & Data Audit Tool")
    parser.add_argument("--data_root", type=str, default=str(DATA_ROOT), help="Dataset root directory")
    args = parser.parse_args()

    process_and_index(args.data_root)
