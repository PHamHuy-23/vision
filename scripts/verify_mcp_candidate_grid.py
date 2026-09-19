"""Verify that real search candidates become one MCP contact-sheet image."""

import asyncio
import base64
import io
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
API_BASE = "http://127.0.0.1:8000"


async def verify() -> None:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.post(
            f"{API_BASE}/api/v1/search",
            json={
                "query": "a person cooking food in a pot",
                "top_k": 12,
                "mode": "semantic",
            },
        )
        response.raise_for_status()
        results = response.json()["results"]
    candidates = [
        {"video_id": item["video_id"], "frame_idx": item["frame_idx"]}
        for item in results
    ]

    server = StdioServerParameters(
        command="python",
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "inspect_candidate_grid",
                {"candidates": candidates, "columns": 4},
            )
            assert not result.isError, result.content
            assert len(result.content) == 1
            content = result.content[0]
            assert content.type == "image"
            assert content.mimeType == "image/jpeg"
            decoded = base64.b64decode(content.data)
            with Image.open(io.BytesIO(decoded)) as sheet:
                assert sheet.size == (1280, 666)
                print(
                    f"candidates={len(candidates)}, size={sheet.size}, "
                    f"jpeg_bytes={len(decoded)}"
                )

            sequence = await session.call_tool(
                "inspect_video_sequence",
                {
                    "video_id": candidates[0]["video_id"],
                    "center_frame": candidates[0]["frame_idx"],
                    "limit": 12,
                    "columns": 4,
                },
            )
            assert not sequence.isError, sequence.content
            sequence_content = sequence.content[0]
            assert sequence_content.type == "image"
            sequence_decoded = base64.b64decode(sequence_content.data)
            with Image.open(io.BytesIO(sequence_decoded)) as sequence_sheet:
                assert sequence_sheet.size == (1280, 666)
                print(
                    f"sequence_size={sequence_sheet.size}, "
                    f"jpeg_bytes={len(sequence_decoded)}"
                )


if __name__ == "__main__":
    asyncio.run(verify())
