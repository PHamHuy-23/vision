import asyncio
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

import sys
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import io
import math
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import httpx
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage, ImageDraw, ImageOps

# Khởi tạo MCP Server
mcp = FastMCP("VideoRetrievalSystem")

API_BASE = "http://127.0.0.1:8000"
DB_PATH = Path(__file__).resolve().with_name("video_index_v2.db")
MAX_EVIDENCE_ROWS_PER_SOURCE = 500


def _build_vision_probe(variant: str) -> bytes:
    """Create a deterministic image for verifying MCP vision end to end."""
    normalized = variant.strip().lower()
    if normalized not in {"probe_a", "probe_b"}:
        raise ValueError("variant must be either 'probe_a' or 'probe_b'")

    canvas = PILImage.new("RGB", (768, 512), "white")
    draw = ImageDraw.Draw(canvas)
    if normalized == "probe_a":
        draw.rectangle((70, 70, 330, 330), fill=(220, 35, 45))
        draw.ellipse((430, 120, 650, 340), fill=(255, 205, 35))
    else:
        draw.polygon([(180, 360), (360, 70), (540, 360)], fill=(30, 105, 220))
        draw.rectangle((585, 120, 685, 340), fill=(35, 180, 90))

    output = io.BytesIO()
    canvas.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue()


@mcp.tool()
def inspect_vision_probe(variant: str = "probe_a") -> Image:
    """
    Return a real image through MCP for a vision-capability test.

    The variant is an opaque test ID: `probe_a` or `probe_b`. Describe the
    visible colors and shapes without inferring them from the ID.
    """
    return Image(data=_build_vision_probe(variant), format="jpeg")


async def _resolve_candidate(
    client: httpx.AsyncClient, candidate: Dict[str, Any]
) -> Dict[str, Any]:
    video_id = str(candidate.get("video_id", "")).strip()
    try:
        frame_idx = int(candidate.get("frame_idx"))
    except (TypeError, ValueError):
        raise ValueError("each candidate requires an integer frame_idx")
    if not video_id:
        raise ValueError("each candidate requires a video_id")

    response = await client.get(
        f"{API_BASE}/api/v1/search/context",
        params={"video_id": video_id, "frame_idx": frame_idx, "limit": 1},
    )
    response.raise_for_status()
    results = response.json().get("results", [])
    if not results:
        raise ValueError(f"frame not found: {video_id}, {frame_idx}")
    return results[0]


async def _download_candidate_image(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    candidate: Dict[str, Any],
) -> tuple[Dict[str, Any], PILImage.Image]:
    image_url = candidate.get("r2_url") or candidate.get("image_path")
    if not image_url:
        raise ValueError(
            f"candidate has no image URL: {candidate['video_id']}, "
            f"{candidate['frame_idx']}"
        )
    allowed_hosts = {
        "pub-63867f61a3cb4f34a8b442399021fbbd.r2.dev",
        "lh3.googleusercontent.com",
    }
    parsed = httpx.URL(image_url)
    if parsed.scheme != "https" or parsed.host not in allowed_hosts:
        raise ValueError(f"unsupported image host: {parsed.host}")

    async with semaphore:
        last_error: Optional[Exception] = None
        for attempt in range(2):
            try:
                response = await client.get(image_url)
                response.raise_for_status()
                break
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt == 0:
                    await asyncio.sleep(0.2)
        else:
            raise ValueError(f"failed to download candidate image: {last_error}")
    if len(response.content) > 8 * 1024 * 1024:
        raise ValueError("candidate image exceeds 8 MB")
    image = PILImage.open(io.BytesIO(response.content)).convert("RGB")
    return candidate, image


def _render_candidate_grid(
    frames: List[tuple[Dict[str, Any], PILImage.Image]], columns: int
) -> bytes:
    tile_width, image_height, label_height = 320, 180, 42
    rows = math.ceil(len(frames) / columns)
    sheet = PILImage.new(
        "RGB", (columns * tile_width, rows * (image_height + label_height)), "#111827"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (candidate, source) in enumerate(frames):
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        tile = ImageOps.pad(
            source, (tile_width, image_height), method=PILImage.Resampling.LANCZOS,
            color="#030712",
        )
        sheet.paste(tile, (x, y))
        label = (
            f"{index + 1:02d}  {candidate['video_id']}  "
            f"F{candidate['frame_idx']}  {candidate.get('timestamp', '')}"
        )
        draw.rectangle((x, y + image_height, x + tile_width, y + image_height + label_height), fill="#111827")
        draw.text((x + 8, y + image_height + 12), label, fill="white")

    output = io.BytesIO()
    sheet.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()


@mcp.tool()
async def inspect_candidate_grid(
    candidates: List[Dict[str, Any]], columns: int = 4
) -> Image:
    """
    Visually inspect 1-20 video candidates in one labeled contact sheet.

    Each candidate must contain only `video_id` and `frame_idx`, copied from
    search results. Labels use the form `NN video_id Fframe timestamp`; cite
    these labels when reporting visual evidence.
    """
    if not 1 <= len(candidates) <= 20:
        raise ValueError("candidates must contain between 1 and 20 items")
    columns = max(2, min(int(columns), 5))
    unique: List[Dict[str, Any]] = []
    seen = set()
    for candidate in candidates:
        key = (str(candidate.get("video_id", "")).strip(), candidate.get("frame_idx"))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)

    timeout = httpx.Timeout(12.0, connect=4.0)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        resolved = await asyncio.gather(
            *(_resolve_candidate(client, candidate) for candidate in unique)
        )
        semaphore = asyncio.Semaphore(6)
        frames = await asyncio.gather(
            *(_download_candidate_image(client, semaphore, candidate) for candidate in resolved)
        )
    return Image(data=_render_candidate_grid(list(frames), columns), format="jpeg")


@mcp.tool()
async def inspect_video_sequence(
    video_id: str, center_frame: int, limit: int = 20, columns: int = 4
) -> Image:
    """
    Visually inspect a chronological sequence around one candidate frame.

    Use this once after a candidate grid when a multi-event query needs proof
    that visible actions belong to the same video. The sheet is ordered from
    earlier to later stored keyframes; cite its labels as direct evidence.
    """
    limit = max(4, min(int(limit), 20))
    columns = max(2, min(int(columns), 5))
    timeout = httpx.Timeout(12.0, connect=4.0)
    limits = httpx.Limits(max_connections=8, max_keepalive_connections=8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, follow_redirects=True) as client:
        response = await client.get(
            f"{API_BASE}/api/v1/video/{video_id}/filmstrip",
            params={
                "anchor_frame": int(center_frame),
                "direction": "around",
                "limit": limit,
            },
        )
        response.raise_for_status()
        candidates = response.json().get("results", [])
        if not candidates:
            raise ValueError(f"no sequence frames found for {video_id}")
        semaphore = asyncio.Semaphore(6)
        frames = await asyncio.gather(
            *(_download_candidate_image(client, semaphore, candidate) for candidate in candidates)
        )
    return Image(data=_render_candidate_grid(list(frames), columns), format="jpeg")

def format_results(results: List[Dict[str, Any]], limit: int = 20) -> str:
    if not results:
        return "Không tìm thấy kết quả phù hợp."
    output = []
    for i, r in enumerate(results[:max(1, min(limit, 20))]):
        score = r.get('score', 0)
        vid = r.get('video_id', 'unknown')
        frame = r.get('frame_idx', 0)
        time_str = r.get('timestamp', '00:00')
        img_url = r.get('r2_url') or r.get('image_path', '')
        
        info = f"• [{i+1}] {vid}, {frame} | Time: {time_str} | Match: {score}%"
        if r.get('ocr_text'):
            info += f" | OCR: {str(r['ocr_text'])[:50]}"
        if r.get('asr_text'):
            info += f" | Voice: {str(r['asr_text'])[:50]}"
        if img_url:
            info += f" | Img: {img_url}"
            
        output.append(info)
    
    return "\n".join(output)

@mcp.tool()
async def search_semantic_video(query: str, top_k: int = 5, video_id: Optional[str] = None) -> str:
    """
    Tìm kiếm khung hình video bằng AI ngữ nghĩa OpenCLIP (Mô tả cảnh vật, con người, hành động, màu sắc, phương tiện...).
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {"query": query, "top_k": top_k, "mode": "semantic"}
            if video_id:
                payload["video_id"] = video_id
            response = await client.post(f"{API_BASE}/api/v1/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return f"Found {data['total_results']} semantic candidates for '{query}':\n" + format_results(data['results'])
    except Exception as e:
        return f"Error connecting to backend: {str(e)}"

@mcp.tool()
async def search_ocr_video(query: str, top_k: int = 5, video_id: Optional[str] = None) -> str:
    """
    Tìm kiếm chữ cái/văn bản xuất hiện trên màn hình video (Biển báo, phụ đề cứng, chữ...).
    Giữ nguyên tiếng Việt, không dịch.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {"query": query, "top_k": top_k, "mode": "ocr"}
            if video_id:
                payload["video_id"] = video_id
            response = await client.post(f"{API_BASE}/api/v1/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return f"Found {data['total_results']} OCR results for '{query}':\n" + format_results(data['results'])
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def search_asr_video(query: str, top_k: int = 5, video_id: Optional[str] = None) -> str:
    """
    Tìm kiếm video bằng lời thoại, giọng nói nhân vật (ASR/Subtitle).
    Giữ nguyên tiếng Việt, không dịch.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            payload = {"query": query, "top_k": top_k, "mode": "asr"}
            if video_id:
                payload["video_id"] = video_id
            response = await client.post(f"{API_BASE}/api/v1/search", json=payload)
            response.raise_for_status()
            data = response.json()
            return f"Found {data['total_results']} ASR results for '{query}':\n" + format_results(data['results'])
    except Exception as e:
        return f"Error: {str(e)}"


@mcp.tool()
async def search_video_evidence(
    terms: List[str], top_videos: int = 5, frames_per_video: int = 4
) -> str:
    """
    Recall videos that contain several independent OCR/ASR evidence terms.

    Use once for a multi-event query with 2-6 short Vietnamese concepts such
    as ingredient or action words. This is candidate recall only, never visual
    proof: inspect the returned video/frame pairs with an image tool.
    """
    started_at = time.perf_counter()
    clean_terms = []
    for term in terms:
        normalized = " ".join(str(term).strip().split())
        if normalized and normalized not in clean_terms:
            clean_terms.append(normalized)
    if not 2 <= len(clean_terms) <= 6:
        raise ValueError("terms must contain 2-6 distinct short concepts")
    top_videos = max(1, min(int(top_videos), 8))
    frames_per_video = max(1, min(int(frames_per_video), 5))

    def search_all_terms():
        term_matches: Dict[str, Dict[str, Dict[str, Any]]] = {}
        skipped_sources = []
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            total_videos = int(
                conn.execute("SELECT COUNT(DISTINCT video_id) FROM keyframes").fetchone()[0]
            )
            for term in clean_terms:
                match_expr = f'"{term.replace(chr(34), chr(34) * 2)}"'
                videos: Dict[str, Dict[str, Any]] = {}
                for mode, table in (("asr", "asr_fts"), ("ocr", "ocr_fts")):
                    source_hits = int(
                        conn.execute(
                            f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?",
                            (match_expr,),
                        ).fetchone()[0]
                    )
                    if source_hits > MAX_EVIDENCE_ROWS_PER_SOURCE:
                        skipped_sources.append((term, mode, source_hits))
                        continue
                    rows = conn.execute(
                        f"""
                        SELECT k.video_id, MIN(k.frame_idx) AS frame_idx,
                               COUNT(*) AS hit_count
                        FROM {table}
                        JOIN keyframes AS k ON k.vector_id = {table}.vector_id
                        WHERE {table} MATCH ?
                        GROUP BY k.video_id
                        """,
                        (match_expr,),
                    ).fetchall()
                    for row in rows:
                        video_id = str(row["video_id"])
                        hit_count = int(row["hit_count"])
                        current = videos.get(video_id)
                        if current is None:
                            videos[video_id] = {
                                "mode": mode,
                                "frame_idx": int(row["frame_idx"]),
                                "hit_count": hit_count,
                                "best_source_hits": hit_count,
                            }
                        else:
                            current["hit_count"] += hit_count
                            if hit_count > current["best_source_hits"]:
                                current["mode"] = mode
                                current["frame_idx"] = int(row["frame_idx"])
                                current["best_source_hits"] = hit_count
                term_matches[term] = videos
        return total_videos, term_matches, skipped_sources

    total_videos, term_matches, skipped_sources = await asyncio.to_thread(search_all_terms)

    matched_terms: Dict[str, set[str]] = defaultdict(set)
    frame_evidence: Dict[str, Dict[tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    hit_counts: Dict[str, int] = defaultdict(int)
    rarity_scores: Dict[str, float] = defaultdict(float)
    for term, videos in term_matches.items():
        # Terms occurring in few videos contribute much more than generic words.
        idf = math.log((total_videos + 1) / (len(videos) + 1)) + 1.0
        for video_id, result in videos.items():
            frame_idx = result["frame_idx"]
            matched_terms[video_id].add(term)
            rarity_scores[video_id] += idf
            hit_counts[video_id] += result["hit_count"]
            key = (term, frame_idx)
            frame_evidence[video_id].setdefault(
                key,
                {
                    "term": term,
                    "mode": result["mode"],
                    "frame_idx": frame_idx,
                    "timestamp": "",
                },
            )

    ranked = sorted(
        matched_terms,
        key=lambda video_id: (
            rarity_scores[video_id],
            len(matched_terms[video_id]),
            math.log1p(hit_counts[video_id]),
        ),
        reverse=True,
    )[:top_videos]
    if not ranked:
        return "No OCR/ASR evidence candidates found."

    def load_timestamps(items: List[tuple[str, int]]) -> Dict[tuple[str, int], str]:
        timestamps = {}
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            for video_id, frame_idx in items:
                row = conn.execute(
                    "SELECT raw_json FROM keyframes WHERE video_id = ? AND frame_idx = ? LIMIT 1",
                    (video_id, frame_idx),
                ).fetchone()
                if not row:
                    continue
                raw = json.loads(row["raw_json"])
                pts = float(raw.get("timestamp", {}).get("pts_time", 0.0))
                timestamps[(video_id, frame_idx)] = (
                    f"{int(pts // 60):02d}:{int(pts % 60):02d} ({pts:.1f}s)"
                )
        return timestamps

    selected_by_video = {}
    timestamp_keys = []
    for video_id in ranked:
        selected = []
        used_terms = set()
        for item in frame_evidence[video_id].values():
            if item["term"] not in used_terms:
                selected.append(item)
                used_terms.add(item["term"])
                timestamp_keys.append((video_id, item["frame_idx"]))
            if len(selected) >= frames_per_video:
                break
        selected_by_video[video_id] = selected
    timestamps = await asyncio.to_thread(load_timestamps, timestamp_keys)

    elapsed = time.perf_counter() - started_at
    lines = [
        f"Metadata evidence v2 completed in {elapsed:.3f}s; candidate recall only; "
        "visually verify every claim:"
    ]
    if skipped_sources:
        skipped = ", ".join(
            f"{term}/{mode} ({count} hits)" for term, mode, count in skipped_sources
        )
        lines.append(f"Ignored overly broad evidence sources: {skipped}.")
    for rank, video_id in enumerate(ranked, 1):
        selected = selected_by_video[video_id]
        frames = "; ".join(
            f"{item['term']}=>{video_id}, {item['frame_idx']} "
            f"({timestamps.get((video_id, item['frame_idx']), '')})"
            for item in selected
        )
        lines.append(
            f"[{rank}] {video_id} matched {len(matched_terms[video_id])}/{len(clean_terms)} "
            f"terms ({', '.join(sorted(matched_terms[video_id]))}); frames: {frames}"
        )
    return "\n".join(lines)

@mcp.tool()
async def get_frame_context(video_id: str, frame_idx: int, limit: int = 5) -> str:
    """
    Xem các khung hình lân cận (Bối cảnh trước/sau) của một frame cụ thể trong video để hiểu diễn biến tiếp theo.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{API_BASE}/api/v1/search/context?video_id={video_id}&frame_idx={frame_idx}&limit={limit}&surrounding=true")
            response.raise_for_status()
            data = response.json()
            return f"Context frames for {video_id} around Frame {frame_idx}:\n" + format_results(data['results'])
    except Exception as e:
        return f"Error: {str(e)}"

@mcp.tool()
async def search_image_by_url(image_url: str, top_k: int = 10, video_id: Optional[str] = None) -> str:
    """
    Tìm kiếm các khung hình video giống với một BỨC ẢNH MẪU. Nhập URL của ảnh mẫu trên mạng vào đây.
    Hệ thống sẽ tải ảnh đó về và so sánh bằng AI.
    """
    try:
        async with httpx.AsyncClient() as client:
            # 1. Tải ảnh từ URL bên ngoài về RAM
            img_resp = await client.get(image_url, timeout=15.0)
            img_resp.raise_for_status()
            image_bytes = img_resp.content

            # 2. Gửi ảnh (multipart/form-data) cho Backend của mình để search
            files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
            data_payload = {"top_k": str(top_k)}
            if video_id:
                data_payload["video_id"] = video_id
                
            response = await client.post(f"{API_BASE}/api/v1/search/image", data=data_payload, files=files, timeout=40.0)
            response.raise_for_status()
            data = response.json()
            return f"Found {data.get('total_results', len(data.get('results', [])))} results for the provided image:\n" + format_results(data['results'])
    except Exception as e:
        return f"Lỗi khi xử lý tìm kiếm ảnh: {str(e)}. Hãy chắc chắn URL ảnh hợp lệ và cho phép tải."

if __name__ == "__main__":
    # Chạy MCP Server trên giao thức stdio
    mcp.run()
