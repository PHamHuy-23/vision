import os
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Import local object JSON files into frame_map_supabase.json (with scores & bboxes)")
    parser.add_argument("--objects-dir", type=str, required=True, help="Path to the directory containing object JSONs (e.g. F:/downloads/database/objects-aic25-b1/objects)")
    parser.add_argument("--map-file", type=str, default="frame_map_supabase.json", help="Path to frame_map_supabase.json")
    parser.add_argument("--min-score", type=float, default=0.1, help="Minimum confidence score to keep")
    
    args = parser.parse_args()
    
    objects_dir = Path(args.objects_dir)
    map_file = Path(args.map_file)
    
    if not objects_dir.exists() or not objects_dir.is_dir():
        print(f"Error: Objects directory '{objects_dir}' does not exist or is not a directory.", flush=True)
        return
        
    if not map_file.exists():
        print(f"Error: Map file '{map_file}' does not exist.", flush=True)
        return
        
    print(f"Loading map file {map_file} (this may take a moment for large files)...", flush=True)
    try:
        with open(map_file, "r", encoding="utf-8") as f:
            frame_map = json.load(f)
    except Exception as e:
        print(f"Error reading map file: {e}")
        return
        
    print(f"Loaded {len(frame_map)} frames. Starting to scan local objects directory...", flush=True)
    
    updated_count = 0
    missing_count = 0
    
    for frame_id, entry in frame_map.items():
        video_id = entry.get("video_id")
        frame_number = entry.get("frame_number")
        
        if not video_id or not frame_number:
            continue
            
        json_filename = f"{frame_number:03d}.json"
        json_path = objects_dir / video_id / json_filename
        
        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as jf:
                    obj_data = json.load(jf)
                    
                scores = obj_data.get("detection_scores", [])
                labels = obj_data.get("detection_class_entities", [])
                boxes = obj_data.get("detection_boxes", [])
                
                valid_labels = set()
                filtered_detections = []
                
                for i in range(len(scores)):
                    score = float(scores[i])
                    if score >= args.min_score:
                        label = labels[i] if i < len(labels) else ""
                        bbox = [float(x) for x in boxes[i]] if i < len(boxes) else []
                        
                        if label:
                            valid_labels.add(label.lower())
                            
                        filtered_detections.append({
                            "label": label,
                            "score": round(score, 4),
                            "bbox": bbox
                        })
                        
                labels_str = ", ".join(sorted(list(valid_labels)))
                
                if "object" not in entry:
                    entry["object"] = {}
                entry["object"]["text"] = labels_str
                entry["object"]["detections"] = filtered_detections
                
                updated_count += 1
                
                if updated_count % 10000 == 0:
                    print(f"Processed {updated_count} object files...", flush=True)
                    
            except Exception as e:
                print(f"Error processing {json_path}: {e}", flush=True)
        else:
            missing_count += 1
            
    print(f"Finished processing. Updated {updated_count} frames. Missing JSONs for {missing_count} frames.", flush=True)
    
    print(f"Saving updated map to {map_file}...", flush=True)
    try:
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(frame_map, f, indent=None)
        print("Save complete!", flush=True)
    except Exception as e:
        print(f"Error saving map file: {e}", flush=True)

if __name__ == "__main__":
    main()
