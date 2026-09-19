import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import time
import subprocess
import threading
import urllib.request
import json
from pathlib import Path

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent

print("=" * 65, flush=True)
print("⏱️ MEASURING SERVER STARTUP TIME FROM COMMAND TO HTTP 200 READY", flush=True)
print("=" * 65, flush=True)

t0 = time.perf_counter()
print(f"[{time.strftime('%H:%M:%S')}] [T0: 0.000s] Launching server process: python -u -m src.main ...", flush=True)

proc = subprocess.Popen(
    [sys.executable, "-u", "-m", "src.main"],
    cwd=str(REPO_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    encoding="utf-8",
    errors="replace"
)

milestones = {}

def read_server_output():
    for line in proc.stdout:
        elapsed = time.perf_counter() - t0
        line_str = line.strip()
        if not line_str:
            continue
        print(f"  [{elapsed:6.2f}s] {line_str}", flush=True)
        
        if "Loaded vector matrix" in line_str and "vectors_loaded" not in milestones:
            milestones["vectors_loaded"] = elapsed
        elif "In-Memory Metadata Cache loaded" in line_str and "metadata_loaded" not in milestones:
            milestones["metadata_loaded"] = elapsed
        elif ("Text encoder" in line_str or "Model loaded successfully" in line_str) and "clip_loaded" not in milestones:
            milestones["clip_loaded"] = elapsed
        elif "Uvicorn running on" in line_str and "uvicorn_started" not in milestones:
            milestones["uvicorn_started"] = elapsed
        elif "Pre-warming Flash" in line_str and "flash_warming" not in milestones:
            milestones["flash_warming"] = elapsed
        elif "Pre-warming Pro" in line_str and "pro_warming" not in milestones:
            milestones["pro_warming"] = elapsed

output_thread = threading.Thread(target=read_server_output, daemon=True)
output_thread.start()

# Poller: checks health check endpoint until 200 OK
health_url = "http://localhost:8000/api/v1/health"
ready = False
poll_start = time.perf_counter()
first_response_data = None

while time.perf_counter() - t0 < 120.0:  # 120s timeout
    try:
        req = urllib.request.Request(health_url, headers={"User-Agent": "Startup-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                t_ready = time.perf_counter()
                total_startup_sec = t_ready - t0
                first_response_data = resp.read().decode('utf-8')
                ready = True
                break
    except Exception:
        pass
    time.sleep(0.05)

if ready:
    print("\n" + "=" * 65, flush=True)
    print(f"🎉 SERVER HOÀN TOÀN KHỞI ĐỘNG THÀNH CÔNG VÀ SẴN SÀNG!", flush=True)
    print(f"⏱️ TỔNG THỜI GIAN KHỞI ĐỘNG (T0 -> HTTP 200 OK): {total_startup_sec:.3f} giây ({total_startup_sec*1000:.0f} ms)", flush=True)
    print("=" * 65, flush=True)
    print("\n📋 Chi tiết các mốc thời gian từ lúc gõ lệnh:", flush=True)
    if "vectors_loaded" in milestones:
        print(f"  1. Tải ma trận 177.321 Vectors & Scene:  +{milestones['vectors_loaded']:.2f}s", flush=True)
    if "metadata_loaded" in milestones:
        print(f"  2. Tải In-Memory Metadata Cache:        +{milestones['metadata_loaded']:.2f}s", flush=True)
    if "clip_loaded" in milestones:
        print(f"  3. Tải OpenCLIP & TorchScript Freeze:   +{milestones['clip_loaded']:.2f}s", flush=True)
    if "uvicorn_started" in milestones:
        print(f"  4. Uvicorn Server lắng nghe port 8000:  +{milestones['uvicorn_started']:.2f}s", flush=True)
    print(f"  👉 Hoàn tất phản hồi HTTP 200 đầu tiên: {total_startup_sec:.2f}s", flush=True)
    print(f"\nDữ liệu phản hồi /api/v1/health:\n{first_response_data}", flush=True)
    print("\n🟢 Server đang tiếp tục chạy ở chế độ nền trên http://localhost:8000 ...", flush=True)
else:
    print(f"\n❌ Timeout sau 120s mà server chưa phản hồi.", flush=True)

# Save startup measurement to a json log
result_file = REPO_ROOT / "startup_time.json"
with open(result_file, "w", encoding="utf-8") as f:
    json.dump({
        "total_startup_seconds": round(total_startup_sec if ready else 120.0, 3),
        "total_startup_ms": round((total_startup_sec if ready else 120.0) * 1000, 1),
        "ready": ready,
        "milestones": milestones,
        "health_response": first_response_data
    }, f, indent=2)

print(f"\nKết quả lưu tại: {result_file}", flush=True)

# Keep server alive
if ready:
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
