import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'

import torch
# torch.set_num_threads(1)

import os
import sys
import ssl
import json
import time
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

# Ensure UTF-8 console output for Windows compatibility
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from .agy_session import AgySession, is_complex_visual_query, session_pool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import asyncio
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "local-user"

from .config import DATA_ROOT, DB_PATH, CONSOLIDATED_VECTORS_PATH, BASE_DIR, HOST, PORT, CORS_ORIGINS
from .sqlite_engine import SQLiteSearchEngine as VectorSearchEngine
from .supabase_service import SupabaseService

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

app = FastAPI(
    title="Video Retrieval & Supabase Google Drive API Backend",
    description="High-performance Video Keyframe Retrieval Backend with Supabase REST API & Google Drive Integration",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_root_path = Path(DATA_ROOT).resolve()
search_engine = VectorSearchEngine(data_root=data_root_path)
supabase_svc = SupabaseService(supabase_url=SUPABASE_URL, supabase_key=SUPABASE_KEY)

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


class SearchSimilarRequest(BaseModel):
    vector_id: int = Field(..., description="Vector ID of the frame to find similar images")
    top_k: int = Field(20, ge=1, le=200, description="Number of top matching results to retrieve")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query in English")
    top_k: int = Field(20, ge=1, le=200, description="Number of top matching results to retrieve")
    video_id: Optional[str] = Field(None, description="Optional Video ID filter constraint")
    mode: Literal["semantic", "ocr", "asr"] = Field("semantic", description="Search mode: semantic, ocr, or asr")

class SearchAllRequest(BaseModel):
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(20, ge=1, le=200)
    video_id: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    print("[Startup] Video Retrieval & Supabase Backend online!", flush=True)
    import asyncio
    
    async def warm_ai_sessions():
        async def warm_one(sid: str, model: str, label: str):
            try:
                print(f"[Startup] Background pre-warming {label} model...", flush=True)
                if sid not in session_pool:
                    session = AgySession(sid, model=model)
                    session_pool[sid] = session
                    await session.start(prewarm=True)
                print(f"[Startup] {label} model ready!", flush=True)
            except Exception as e:
                print(f"[Startup] {label} pre-warm notice: {e}", flush=True)

        await asyncio.gather(
            warm_one("local-flash", "flash", "Flash"),
            warm_one("local-pro", "pro", "Pro"),
        )

    asyncio.create_task(warm_ai_sessions())
    print("[Startup] Server ready to accept HTTP traffic!", flush=True)


@app.get("/")
def read_root():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "Video Retrieval System & Supabase Backend API",
        "version": "2.1.0",
        "status": "online",
        "supabase_configured": supabase_svc.is_configured,
        "docs": "/docs"
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "supabase_connected": supabase_svc.is_configured,
        "database_connected": DB_PATH.exists(),
        "vector_matrix_loaded": CONSOLIDATED_VECTORS_PATH.exists(),
        "total_keyframes": 177321
    }


# Load video metadata into memory
video_metadata_cache = {}
try:
    with open("video_drive_metadata.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        if "videos" in data:
            video_metadata_cache = data["videos"]
except Exception as e:
    print(f"Warning: Could not load video_drive_metadata.json: {e}")

@app.get("/api/v1/video/{video_id}")
def get_video_info(video_id: str):
    info = video_metadata_cache.get(video_id)
    if not info:
        raise HTTPException(status_code=404, detail="Video metadata not found")
    return {
        "video_id": video_id,
        "drive_file_id": info.get("drive_file_id"),
        "drive_url": info.get("drive_url")
    }


# ====================================================================
# SUPABASE & GOOGLE DRIVE ENDPOINTS
# ====================================================================

@app.get("/api/v1/supabase/video/{video_id}")
def get_supabase_video(video_id: str):
    """Retrieve video metadata & Google Drive File IDs directly from Supabase."""
    video = supabase_svc.get_video(video_id)
    if not video:
        raise HTTPException(status_code=404, detail=f"Video ID '{video_id}' not found in Supabase")
    return video


@app.get("/api/v1/supabase/frame/{frame_id}")
def get_supabase_frame(frame_id: str):
    """Retrieve single frame metadata & Google Drive CDN Image URL directly from Supabase."""
    frame = supabase_svc.get_frame(frame_id)
    if not frame:
        raise HTTPException(status_code=404, detail=f"Frame ID '{frame_id}' not found in Supabase")
    return frame


@app.get("/api/v1/supabase/video/{video_id}/frames")
def get_supabase_video_frames(video_id: str, limit: int = Query(500, ge=1, le=2000)):
    """Retrieve all keyframes of a video ordered by frame_number from Supabase."""
    frames = supabase_svc.get_frames_by_video(video_id, limit=limit)
    return {
        "video_id": video_id,
        "total_frames": len(frames),
        "frames": frames
    }


import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from fastapi.responses import Response

# Limit concurrent Google Drive requests to prevent 403 Rate Limits and thread blocking
drive_semaphore = asyncio.Semaphore(8)
drive_executor = ThreadPoolExecutor(max_workers=8)

@lru_cache(maxsize=1000)
def fetch_drive_file_cached(file_id: str) -> tuple[bytes, str]:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req, context=ssl_context, timeout=20)
    return res.read(), res.headers.get('Content-Type', 'application/octet-stream')

async def fetch_drive_file_async(file_id: str):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(drive_executor, fetch_drive_file_cached, file_id)

@app.get("/api/v1/drive/proxy/{file_id}")
async def proxy_google_drive_file(file_id: str):
    """Proxy any file content from Google Drive with in-memory LRU caching."""
    async with drive_semaphore:
        for attempt in range(3):
            try:
                content, mime_type = await fetch_drive_file_async(file_id)
                return Response(content=content, media_type=mime_type)
            except Exception as e:
                if attempt == 2:
                    raise HTTPException(status_code=500, detail=f"Failed fetching Google Drive File {file_id}: {str(e)}")
                await asyncio.sleep(0.5)


# ====================================================================
# VECTOR SEARCH ENDPOINTS
# ====================================================================

@app.get("/api/v1/search/context")
def search_context(video_id: str, frame_idx: int, limit: int = 20, surrounding: bool = False):
    results = search_engine.search_context(video_id, frame_idx, limit, surrounding=surrounding)
    return {"status": "success", "results": results}

@app.get("/api/v1/video/{video_id}/frames")
def video_frames(video_id: str, start_frame: int, end_frame: int, limit: int = 80):
    """Return the stored keyframes in a bounded range for smooth filmstrip prefetching."""
    if end_frame < start_frame:
        start_frame, end_frame = end_frame, start_frame
    limit = max(1, min(limit, 200))
    results = search_engine.search_frame_range(video_id, start_frame, end_frame, limit)
    return {"status": "success", "video_id": video_id, "start_frame": start_frame, "end_frame": end_frame, "results": results}

@app.get("/api/v1/video/{video_id}/filmstrip")
def video_filmstrip(video_id: str, anchor_frame: int, direction: str = "around", limit: int = 40):
    if direction not in {"around", "before", "after"}:
        raise HTTPException(status_code=400, detail="direction must be around, before, or after")
    results = search_engine.search_frame_page(video_id, anchor_frame, direction, limit)
    return {"status": "success", "video_id": video_id, "anchor_frame": anchor_frame, "direction": direction, "results": results}

@app.get("/api/v1/search/interval")
def search_interval(video_id: str, start_time: float, end_time: float, limit: int = 200):
    results = search_engine.search_interval(video_id, start_time, end_time, limit)
    return {"status": "success", "results": results}

@app.get("/api/v1/video/{video_id}/convert_time")
def convert_time_to_frame(video_id: str, time_sec: float, fps: float = 25.0):
    # Fetch actual FPS from local JSON map to be portable across machines
    actual_fps = fps
    try:
        json_path = BASE_DIR / "video_fps_map.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                fps_map = json.load(f)
                if video_id in fps_map:
                    actual_fps = fps_map[video_id]
    except Exception as e:
        print(f"Warning: Could not read local FPS map: {e}")

    frame_idx = round(time_sec * actual_fps)
    return {"status": "success", "video_id": video_id, "time_sec": time_sec, "frame_idx": frame_idx, "fps": actual_fps}

@app.post("/api/v1/search/similar")
def search_similar(req: SearchSimilarRequest):
    results = search_engine.search(
        query_text="",
        query_vector_id=req.vector_id,
        top_k=req.top_k
    )
    return {"status": "success", "results": results}

@app.post("/api/v1/search/image")
async def search_by_image(file: UploadFile = File(...), top_k: int = Form(50), video_id: Optional[str] = Form(None)):
    image_bytes = await file.read()
    results = search_engine.search_by_image(image_bytes, top_k=top_k, video_id_filter=video_id)
    return {
        "status": "success",
        "total_results": len(results),
        "results": results
    }

@app.post("/api/v1/chat")
async def chat_endpoint(req: ChatRequest):
    # Model Routing Logic
    is_complex = is_complex_visual_query(req.message)
    
    # Force use pre-warmed routed session instead of frontend's static ID
    sid = "local-pro" if is_complex else "local-flash"
    
    if sid not in session_pool:
        session = AgySession(sid, model="pro" if is_complex else "flash")
        session_pool[sid] = session
    else:
        session = session_pool[sid]

    async def gen():
        try:
            async for chunk in session.send_message(req.message):
                yield chunk
        except Exception as e:
            yield f"data: [ERROR] Lỗi hệ thống: {str(e)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

@app.post("/api/v1/search")
def search_keyframes(req: SearchRequest):
    import time
    start_time = time.time()
    
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    # Fast Local/Cached Auto-translate Vietnamese to English for semantic search
    if req.mode in ["semantic", "smart"]:
        try:
            from .fast_translator import fast_translator
            translated = fast_translator.translate(req.query)
            if translated != req.query:
                print(f"[FastTranslator] '{req.query}' -> '{translated}'", flush=True)
                req.query = translated
        except Exception as e:
            print(f"[FastTranslator] Translation warning: {e}", flush=True)

    if req.mode == "ocr":
        results = search_engine.exact_ocr_search(
            query_text=req.query,
            top_k=req.top_k,
            video_id_filter=req.video_id
        )
        print(f"[Search API] OCR search took {time.time() - start_time:.3f}s", flush=True)
    elif req.mode == "asr":
        results = search_engine.exact_asr_search(
            query_text=req.query,
            top_k=req.top_k,
            video_id_filter=req.video_id
        )
        print(f"[Search API] ASR search took {time.time() - start_time:.3f}s", flush=True)
    else:
        results = search_engine.search(
            query_text=req.query,
            top_k=req.top_k,
            video_id_filter=req.video_id
        )

    return {
        "query": req.query,
        "mode": req.mode,
        "total_results": len(results),
        "results": results
    }

def _search_one_mode(query: str, mode: str, top_k: int, video_id: Optional[str]):
    """Run one independent retrieval branch for the all-modes endpoint."""
    if mode == "semantic":
        return search_engine.search(query_text=query, top_k=top_k, video_id_filter=video_id)
    if mode == "ocr":
        return search_engine.exact_ocr_search(query, top_k=top_k, video_id_filter=video_id)
    return search_engine.exact_asr_search(query, top_k=top_k, video_id_filter=video_id)

@app.post("/api/v1/search/all")
async def search_all_modes(req: SearchAllRequest):
    """Search CLIP, OCR and ASR concurrently; the UI renders each branch separately."""
    import asyncio
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    async def run(mode: str):
        started = time.perf_counter()
        results = await asyncio.to_thread(
            _search_one_mode, req.query, mode, req.top_k, req.video_id
        )
        return mode, {
            "status": "complete",
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
            "total_results": len(results),
            "results": results,
        }

    branches = await asyncio.gather(*(run(mode) for mode in ("semantic", "ocr", "asr")))
    return {"query": req.query, "results": dict(branches)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

