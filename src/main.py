import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

import os
import sys
import ssl
import json
import urllib.request
from pathlib import Path
from typing import Optional, List, Dict, Any

# Ensure UTF-8 console output for Windows compatibility
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import DATA_ROOT, DB_PATH, CONSOLIDATED_VECTORS_PATH, BASE_DIR, HOST, PORT, CORS_ORIGINS
from .search_engine import VectorSearchEngine
from .db import IndexDatabase
from .supabase_service import SupabaseService

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
GDRIVE_API_KEY = os.getenv("GDRIVE_API_KEY", "AIzaSyCs_2-bm7Duz_tctK9cvtUhTJm7vtIbmEE")

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
db = IndexDatabase(DB_PATH)
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
    mode: str = Field("semantic", description="Search mode: semantic, keyword, or smart")
    enable_rerank: bool = Field(False, description="Enable Gemini Vision re-ranking (slower but more accurate)")


@app.on_event("startup")
async def startup_event():
    print("[Startup] Video Retrieval & Supabase Backend online!", flush=True)


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
        "total_keyframes": db.get_all_count() or 173605
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


from functools import lru_cache
from fastapi import Response

@lru_cache(maxsize=1000)
def fetch_drive_file_cached(file_id: str) -> tuple[bytes, str]:
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    res = urllib.request.urlopen(req, context=ssl_context, timeout=20)
    return res.read(), res.headers.get('Content-Type', 'application/octet-stream')

@app.get("/api/v1/drive/proxy/{file_id}")
def proxy_google_drive_file(file_id: str):
    """Proxy any file content from Google Drive with in-memory LRU caching."""
    try:
        content, mime_type = fetch_drive_file_cached(file_id)
        return Response(content=content, media_type=mime_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed fetching Google Drive File {file_id}: {str(e)}")


# ====================================================================
# VECTOR SEARCH ENDPOINTS
# ====================================================================

@app.get("/api/v1/search/context")
def search_context(video_id: str, frame_idx: int, limit: int = 20):
    results = search_engine.search_context(video_id, frame_idx, limit)
    return {"status": "success", "results": results}

@app.post("/api/v1/search/similar")
def search_similar(req: SearchSimilarRequest):
    results = search_engine.search(
        query_text="",
        query_vector_id=req.vector_id,
        top_k=req.top_k
    )
    return {"status": "success", "results": results}

@app.post("/api/v1/search")
def search_keyframes(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    if "->" in req.query:
        queries = [q.strip() for q in req.query.split("->") if q.strip()]
        results = search_engine.temporal_search(
            queries=queries,
            top_k=req.top_k
        )
    elif req.mode == "smart":
        results = search_engine.smart_search(
            query_text=req.query,
            top_k=req.top_k,
            video_id_filter=req.video_id,
            enable_rerank=req.enable_rerank
        )
    elif req.mode == "keyword":
        results = search_engine.keyword_search(
            query_text=req.query,
            top_k=req.top_k,
            video_id_filter=req.video_id
        )
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=False)

