"""Run one end-to-end multi-event query through the persistent Agy session."""

import asyncio
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agy_session import AgySession


QUERY = """Một chén trứng gà đã đánh tan được đổ vào nồi nước súp đang nấu.
Trong nồi có các nguyên liệu được cắt thành sợi, gồm nấm và măng.
Người đầu bếp dùng dao cắt đậu hũ thành từng miếng nhỏ ngay phía trên nồi.
Đậu hũ được cho trực tiếp vào nồi rồi các nguyên liệu được khuấy đều với nhau."""
EXPECTED_VIDEO = "L26_V424"


async def verify() -> None:
    session = AgySession("visual-orchestrator-smoke", model="pro")
    started = time.perf_counter()
    await session.start()
    ready = time.perf_counter()
    chunks = []
    try:
        async for chunk in session.send_message(QUERY):
            chunks.append(chunk)
    finally:
        await session.close()
    finished = time.perf_counter()
    output = "".join(chunks)
    assert "[DONE]" in output
    assert "[ERROR]" not in output, output
    assert EXPECTED_VIDEO in output, output
    print(f"startup_seconds={ready - started:.2f}", flush=True)
    print(f"turn_seconds={finished - ready:.2f}", flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    asyncio.run(asyncio.wait_for(verify(), timeout=120.0))
