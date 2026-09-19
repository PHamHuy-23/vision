"""Verify that the video MCP server transports actual image content."""

import asyncio
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parent.parent


async def verify() -> None:
    server = StdioServerParameters(
        command="python",
        args=[str(PROJECT_ROOT / "mcp_server.py")],
        cwd=PROJECT_ROOT,
    )
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert "inspect_vision_probe" in names

            for variant in ("probe_a", "probe_b"):
                result = await session.call_tool(
                    "inspect_vision_probe", {"variant": variant}
                )
                assert not result.isError
                assert len(result.content) == 1
                image = result.content[0]
                assert image.type == "image"
                assert image.mimeType == "image/jpeg"
                assert len(image.data) > 1_000
                print(
                    f"{variant}: type={image.type}, "
                    f"mime={image.mimeType}, base64_chars={len(image.data)}"
                )


if __name__ == "__main__":
    asyncio.run(verify())
