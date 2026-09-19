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

import httpx
from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP, Image
from PIL import Image as PILImage, ImageDraw

# Khởi tạo MCP Server
mcp = FastMCP("VideoRetrievalSystem")

API_BASE = "http://127.0.0.1:8000"


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

def format_results(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "Không tìm thấy kết quả phù hợp."
    output = []
    for i, r in enumerate(results[:5]):
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
