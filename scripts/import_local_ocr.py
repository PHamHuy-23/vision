import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Import local OCR JSON files into frame_map_supabase.json")
    parser.add_argument("--ocr-dir", type=str, default="G:/OCR_Batches_Zipped", help="Path to the directory containing OCR batches (e.g. G:/OCR_Batches_Zipped)")
    parser.add_argument("--map-file", type=str, default="frame_map_supabase.json", help="Path to frame_map_supabase.json")
    
    args = parser.parse_args()
    
    ocr_dir = Path(args.ocr_dir)
    map_file = Path(args.map_file)
    
    if not ocr_dir.exists() or not ocr_dir.is_dir():
        print(f"Error: OCR directory '{ocr_dir}' does not exist or is not a directory.")
        return
        
    if not map_file.exists():
        print(f"Error: Map file '{map_file}' does not exist.")
        return
        
    print(f"Loading map file {map_file} (this may take a moment)...")
    try:
        with open(map_file, "r", encoding="utf-8") as f:
            frame_map = json.load(f)
    except Exception as e:
        print(f"Error reading map file: {e}")
        return
        
    updated_count = 0
    missing_ocr = 0
    
    total_frames = len(frame_map)
    processed = 0
    
    print(f"Loaded {total_frames} frames. Beginning OCR data extraction...")
    
    for frame_id, entry in frame_map.items():
        processed += 1
        if processed % 10000 == 0:
            print(f"Processed {processed}/{total_frames} frames...", flush=True)
            
        ocr_meta = entry.get("ocr", {})
        if not ocr_meta:
            continue
            
        json_meta = ocr_meta.get("json") or {}
        rel_path = json_meta.get("rel_path")
        
        if not rel_path:
            continue
            
        # Construct absolute path
        # rel_path looks like "Keyframes_L21/keyframes/ocr_L21_V001/001_ocr.json"
        json_path = ocr_dir / rel_path
        
        if not json_path.exists():
            missing_ocr += 1
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                ocr_data = json.load(jf)
                
            full_text = ocr_data.get("full_text", "")
            # Clean text (replace newlines with spaces)
            cleaned_text = " ".join(full_text.split())
            
            detections = ocr_data.get("detections", [])
            filtered_detections = []
            
            for det in detections:
                conf = float(det.get("confidence", 0.0))
                # Skip absolute garbage text
                if conf >= 0.1:
                    filtered_detections.append({
                        "text": det.get("text", ""),
                        "conf": round(conf, 4),
                        "bbox": det.get("bbox", [])
                    })
                    
            entry["ocr"]["text"] = cleaned_text
            entry["ocr"]["detections"] = filtered_detections
            updated_count += 1
            
        except Exception as e:
            print(f"Error processing {json_path}: {e}", flush=True)
            
    print(f"Finished processing. Updated OCR for {updated_count} frames. Missing OCR files for {missing_ocr} frames.", flush=True)
    
    print(f"Saving updated map to {map_file}...")
    try:
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(frame_map, f, indent=None)
        print("Save complete!")
    except Exception as e:
        print(f"Error saving map file: {e}")

if __name__ == "__main__":
    main()
