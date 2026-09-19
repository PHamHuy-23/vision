"""Benchmark evidence recall for known videos and broad common terms."""

import asyncio
import sys
import time
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
            cases = [
                ("soup", ["trứng", "đậu hũ", "nấm", "măng", "súp", "nước dùng"], "L26_V424"),
                ("buffalo", ["đua trâu", "trâu nước", "trâu trắng", "ruộng bùn"], "L21_V005"),
                ("common", ["xe đạp", "nước", "mặt", "áo đỏ", "nón"], None),
            ]
            for name, terms, expected_video in cases:
                started = time.perf_counter()
                result = await session.call_tool(
                    "search_video_evidence",
                    {"terms": terms, "top_videos": 5, "frames_per_video": 4},
                )
                elapsed = time.perf_counter() - started
                assert not result.isError, result.content
                output = result.content[0].text
                if expected_video:
                    assert expected_video in output, output
                # Includes MCP stdio/schema refresh overhead, not only SQLite time.
                assert elapsed < 12.0, f"{name} evidence recall took {elapsed:.2f}s"
                print(f"[{name}] {elapsed:.3f}s")
                print(output)


if __name__ == "__main__":
    asyncio.run(verify())
