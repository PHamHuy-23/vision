"""Verify multi-term evidence recall includes the known soup video."""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


async def verify() -> None:
    server = StdioServerParameters(
        command="python",
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_video_evidence",
                {
                    "terms": ["trứng", "đậu hũ", "nấm", "măng", "súp", "nước dùng"],
                    "top_videos": 5,
                    "frames_per_video": 4,
                },
            )
            assert not result.isError, result.content
            output = result.content[0].text
            assert "L26_V424" in output, output
            print(output)


if __name__ == "__main__":
    asyncio.run(verify())
