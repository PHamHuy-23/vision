"""Run one end-to-end multi-event query through the persistent Agy session."""

import asyncio
import argparse
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agy_session import AgySession


QUERY = """Một chén trứng gà đã đánh tan được đổ vào nồi nước súp đang nấu.
Trong nồi có các nguyên liệu được cắt thành sợi, gồm nấm và măng.
Người đầu bếp dùng dao cắt đậu hũ thành từng miếng nhỏ ngay phía trên nồi.
Đậu hũ được cho trực tiếp vào nồi rồi các nguyên liệu được khuấy đều với nhau."""
EXPECTED_VIDEO = "L26_V424"
BUFFALO_QUERY = """Trong clip có cảnh 3 cặp trâu nước đang được điều khiển chạy đua trên ruộng bùn ở 1 quốc gia Đông Nam Á.
Sau đó có cảnh 1 cặp trâu màu trắng (màu sáng) chạy đua."""
BENCHMARKS = {
    "soup": (QUERY, [EXPECTED_VIDEO, "5491"]),
    "buffalo": (BUFFALO_QUERY, ["L21_V005", "24326"]),
}


async def verify(query: str, expected_tokens: list[str] | None = None) -> None:
    session = AgySession("visual-orchestrator-smoke", model="pro")
    started = time.perf_counter()
    await session.start(prewarm=True)
    ready = time.perf_counter()
    chunks = []
    try:
        async for chunk in session.send_message(query):
            chunks.append(chunk)
    finally:
        await session.close()
    finished = time.perf_counter()
    output = "".join(chunks)
    assert "[DONE]" in output
    assert "[ERROR]" not in output, output
    for token in expected_tokens or []:
        assert token in output, output
    print(f"startup_seconds={ready - started:.2f}", flush=True)
    print(f"turn_seconds={finished - ready:.2f}", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(BENCHMARKS), default="soup")
    parser.add_argument("--query", help="Custom benchmark query")
    parser.add_argument("--expected-video", help="Optional expected VideoID")
    parser.add_argument("--expected-frame", help="Optional expected FrameIdx")
    args = parser.parse_args()
    benchmark_query, expected_tokens = BENCHMARKS[args.case]
    if args.query:
        benchmark_query = args.query
        expected_tokens = [value for value in (args.expected_video, args.expected_frame) if value]
    asyncio.run(
        asyncio.wait_for(
            verify(benchmark_query, expected_tokens=expected_tokens), timeout=120.0
        )
    )
