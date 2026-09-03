import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Import local ASR JSON files into frame_map_supabase.json using Time-Window Anchoring")
    parser.add_argument("--asr-dir", type=str, required=True, help="Path to the directory containing ASR JSONs (e.g. G:/Desktop/subtitle/subtitles/json)")
    parser.add_argument("--map-file", type=str, default="frame_map_supabase.json", help="Path to frame_map_supabase.json")
    parser.add_argument("--window", type=float, default=2.0, help="Time window padding in seconds (e.g. +/- 2.0s)")
    
    args = parser.parse_args()
    
    asr_dir = Path(args.asr_dir)
    map_file = Path(args.map_file)
    
    if not asr_dir.exists() or not asr_dir.is_dir():
        print(f"Error: ASR directory '{asr_dir}' does not exist or is not a directory.")
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
        
    # Group frames by video_id to optimize search
    frames_by_video = defaultdict(list)
    for frame_id, entry in frame_map.items():
        vid = entry.get("video_id")
        ts = entry.get("timestamp", {})
        pts = float(ts.get("pts_time", 0.0))
        if vid:
            frames_by_video[vid].append((frame_id, pts, entry))
            
    print(f"Loaded {len(frame_map)} frames across {len(frames_by_video)} videos.")
    
    updated_count = 0
    missing_asr = 0
    processed_videos = 0
    
    # Process each video
    for vid, frames in frames_by_video.items():
        json_path = asr_dir / f"{vid}_sub.json"
        
        if not json_path.exists():
            missing_asr += 1
            continue
            
        try:
            with open(json_path, "r", encoding="utf-8") as jf:
                asr_data = json.load(jf)
                
            transcripts = asr_data.get("transcript", [])
            
            # Sort frames by pts_time for binary search / easy closest match
            frames.sort(key=lambda x: x[1])
            
            # Dictionary to accumulate texts for each frame_id
            frame_texts = defaultdict(list)
            
            for segment in transcripts:
                seg_start = float(segment.get("start", 0.0))
                seg_dur = float(segment.get("duration", 0.0))
                seg_end = seg_start + seg_dur
                text = segment.get("text", "").strip()
                
                if not text:
                    continue
                    
                # Find frames within window
                matched = False
                closest_frame = None
                min_dist = float('inf')
                
                for frame_id, pts, entry in frames:
                    # Overlap check with padding
                    if (pts + args.window) >= seg_start and (pts - args.window) <= seg_end:
                        frame_texts[frame_id].append(text)
                        matched = True
                    
                    # Track closest frame in case of no overlap
                    seg_mid = (seg_start + seg_end) / 2
                    dist = abs(pts - seg_mid)
                    if dist < min_dist:
                        min_dist = dist
                        closest_frame = frame_id
                        
                # Lossless Guarantee: If it fell in a gap, attach to the absolute closest keyframe
                if not matched and closest_frame is not None:
                    frame_texts[closest_frame].append(text)
                    
            # Write accumulated texts back to the entries
            for frame_id, texts in frame_texts.items():
                combined_text = " ".join(texts)
                # We need to find the entry object
                for fid, pts, entry in frames:
                    if fid == frame_id:
                        if "asr" not in entry:
                            entry["asr"] = {}
                        entry["asr"]["text"] = combined_text
                        updated_count += 1
                        break
                        
        except Exception as e:
            print(f"Error processing {json_path}: {e}", flush=True)
            
        processed_videos += 1
        if processed_videos % 50 == 0:
            print(f"Processed {processed_videos}/{len(frames_by_video)} videos...", flush=True)
            
    print(f"Finished processing. Updated ASR for {updated_count} frames. Missing ASR for {missing_asr} videos.", flush=True)
    
    print(f"Saving updated map to {map_file}...")
    try:
        # Write back to file
        with open(map_file, "w", encoding="utf-8") as f:
            json.dump(frame_map, f, indent=None)
        print("Save complete!")
    except Exception as e:
        print(f"Error saving map file: {e}")

if __name__ == "__main__":
    main()
