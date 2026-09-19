import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['HF_HUB_OFFLINE'] = '1'

import sys
import time
import json
import urllib.request
from pathlib import Path
import numpy as np

# Ensure UTF-8 console output
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
API_BASE = "http://localhost:8000/api/v1"

print("=" * 70, flush=True)
print("🚀 HỆ THỐNG ĐO TỐC ĐỘ AI (AI RETRIEVAL BENCHMARK SUITE)", flush=True)
print("=" * 70, flush=True)

# 1. Health check to ensure server is ready
try:
    with urllib.request.urlopen(f"{API_BASE}/health", timeout=5) as resp:
        health_data = json.loads(resp.read().decode('utf-8'))
        print(f"🟢 Server status: {health_data['status']} | Total keyframes: {health_data['total_keyframes']:,}\n", flush=True)
except Exception as e:
    print(f"❌ Server không phản hồi: {e}")
    sys.exit(1)

# Test queries
test_queries = [
    {"vi": "người lái xe máy trên đường", "type": "Phương tiện / Giao thông"},
    {"vi": "cảnh sát giao thông đang làm việc", "type": "Hành động / Đồng phục"},
    {"vi": "phỏng vấn trong trường quay truyền hình", "type": "Không gian / Studio"},
    {"vi": "người đang nấu ăn trong bếp", "type": "Hoạt cảnh sinh hoạt"},
    {"vi": "máy bay hạ cánh xuống phi trường", "type": "Hàng không / Cảnh quay xa"}
]

def call_search_api(query: str, mode: str = "semantic", top_k: int = 20):
    payload = json.dumps({
        "query": query,
        "mode": mode,
        "top_k": top_k
    }).encode("utf-8")
    
    req = urllib.request.Request(
        f"{API_BASE}/search",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        duration = (time.perf_counter() - t0) * 1000  # ms
        data = json.loads(resp.read().decode("utf-8"))
    return duration, data

# --- TEST 1: END-TO-END SEARCH LATENCY (COLD & HOT) ---
print("--- [1] ĐO TỐC ĐỘ END-TO-END API TÌM KIẾM SEMANTIC (AI VECTOR SEARCH) ---", flush=True)
print(f"{'STT':<4} | {'Query (Tiếng Việt)':<38} | {'Cold Run (Lần 1)':<18} | {'Hot/Cached (Lần 2)':<18} | {'Số kết quả':<10}", flush=True)
print("-" * 96, flush=True)

search_results_summary = []

for i, q in enumerate(test_queries, 1):
    q_vi = q["vi"]
    
    # Cold run
    dur_cold, data_cold = call_search_api(q_vi, mode="semantic", top_k=20)
    
    # Hot run (cached translation + warm CPU cache)
    dur_hot, data_hot = call_search_api(q_vi, mode="semantic", top_k=20)
    
    total_res = data_hot.get("total_results", 0)
    top_score = data_hot["results"][0]["score"] if data_hot.get("results") else 0
    top_video = data_hot["results"][0]["video_id"] if data_hot.get("results") else "N/A"
    
    print(f"{i:<4} | {q_vi:<38} | {dur_cold:8.1f} ms        | {dur_hot:8.1f} ms        | {total_res:<10}", flush=True)
    
    search_results_summary.append({
        "query": q_vi,
        "cold_ms": dur_cold,
        "hot_ms": dur_hot,
        "total_results": total_res,
        "top_1_video": top_video,
        "top_1_score": top_score
    })

avg_cold = np.mean([s["cold_ms"] for s in search_results_summary])
avg_hot = np.mean([s["hot_ms"] for s in search_results_summary])
print("-" * 96, flush=True)
print(f"👉 TRUNG BÌNH: Lần đầu (Cold): {avg_cold:.1f} ms ({avg_cold/1000:.2f}s)  |  Lần sau (Hot Cache): {avg_hot:.1f} ms ({avg_hot/1000:.2f}s)\n", flush=True)


# --- TEST 2: MICRO-BENCHMARKING BÊN TRONG CÁC TẦNG AI ---
print("--- [2] PHÂN RÃ CHI TIẾT TỪNG GIAI ĐOẠN CỦA PIPELINE AI (MICRO-BENCHMARK) ---", flush=True)
from src.fast_translator import fast_translator
from src.sqlite_engine import SQLiteSearchEngine
from src.config import DATA_ROOT

engine = SQLiteSearchEngine(data_root=DATA_ROOT)

# Sample query to benchmark stages
sample_query = "cảnh sát giao thông đang làm nhiệm vụ trên phố"

# Stage 1: Fast Translation
# 1a. Cold translation (simulate unseen query)
t0 = time.perf_counter()
trans_text = fast_translator.translate(sample_query)
t_trans = (time.perf_counter() - t0) * 1000

# 1b. In-memory cached translation
t0 = time.perf_counter()
trans_cached = fast_translator.translate(sample_query)
t_trans_cache = (time.perf_counter() - t0) * 1000

# Stage 2: OpenCLIP Text Encoding (AI ViT-B-32 CPU Inference)
t0 = time.perf_counter()
query_vec = engine.encode_text(trans_text)
t_encode = (time.perf_counter() - t0) * 1000

# Stage 3: Vector Similarity (177,321 vectors dot product)
t0 = time.perf_counter()
scores_all = np.dot(engine.vectors, query_vec)
t_dot = (time.perf_counter() - t0) * 1000

# Stage 4: Top-K Selection (Argpartition + Sort for 20 candidates)
t0 = time.perf_counter()
top_k = 20
top_indices = np.argpartition(scores_all, -top_k)[-top_k:]
top_indices = top_indices[np.argsort(scores_all[top_indices])[::-1]]
scores = scores_all[top_indices]
t_topk = (time.perf_counter() - t0) * 1000

# Stage 5: In-Memory Metadata Cache Retrieval (20 items)
t0 = time.perf_counter()
results = []
for idx, score in zip(top_indices, scores):
    meta = engine._get_metadata_item(int(idx))
    if meta:
        item = meta.copy()
        item["score"] = float(round(score * 100, 2))
        results.append(item)
t_meta = (time.perf_counter() - t0) * 1000

print(f"Query thử nghiệm: \"{sample_query}\" -> \"{trans_text}\"\n")
print(f"  1. 🌐 AI Translation (CTranslate2 INT8):           {t_trans:7.2f} ms (Lần đầu: {t_trans:.1f}ms | Cache hit: {t_trans_cache:.3f}ms)")
print(f"  2. 🧠 AI CLIP Text Encoder (ViT-B-32 PyTorch CPU): {t_encode:7.2f} ms")
print(f"  3. ⚡ Ma trận tương đồng 177.321 Vectors (NumPy BLAS): {t_dot:7.2f} ms")
print(f"  4. 🎯 Lọc Top-20 Candidates (Argpartition O(N)):   {t_topk:7.2f} ms")
print(f"  5. 💾 Trích xuất In-Memory Metadata (20 items):    {t_meta:7.2f} ms")
total_core_pipeline = t_trans_cache + t_encode + t_dot + t_topk + t_meta
print(f"  -------------------------------------------------------------")
print(f"  👉 TỔNG THỜI GIAN TÍNH TOÁN CỦA THUẬT TOÁN AI:      {total_core_pipeline:7.2f} ms ({total_core_pipeline/1000:.3f}s)\n")


# --- TEST 3: SO SÁNH CÁC CHẾ ĐỘ TÌM KIẾM KHÁC (OCR & ASR FTS5) ---
print("--- [3] SO SÁNH TỐC ĐỘ CÁC CHẾ ĐỘ TÌM KIẾM KHÁC (FTS5 TRIGRAM) ---", flush=True)
dur_ocr, data_ocr = call_search_api("cảnh sát", mode="ocr", top_k=20)
dur_asr, data_asr = call_search_api("xin chào", mode="asr", top_k=20)
print(f"  • OCR Search (FTS5 Trigram): {dur_ocr:6.1f} ms  (Tìm thấy {data_ocr.get('total_results', 0)} keyframes có chữ 'cảnh sát')")
print(f"  • ASR Search (FTS5 Trigram): {dur_asr:6.1f} ms  (Tìm thấy {data_asr.get('total_results', 0)} keyframes có giọng nói 'xin chào')\n")

# Save benchmark results to JSON file
bench_file = REPO_ROOT / "ai_search_benchmark.json"
with open(bench_file, "w", encoding="utf-8") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "average_cold_ms": round(avg_cold, 2),
        "average_hot_ms": round(avg_hot, 2),
        "pipeline_breakdown_ms": {
            "ai_translation_ctranslate2": round(t_trans, 2),
            "ai_translation_cache": round(t_trans_cache, 4),
            "ai_clip_encoder": round(t_encode, 2),
            "vector_similarity_177k": round(t_dot, 2),
            "topk_selection": round(t_topk, 2),
            "metadata_retrieval": round(t_meta, 2),
            "total_ai_core_cached": round(total_core_pipeline, 2)
        },
        "query_benchmarks": search_results_summary,
        "ocr_latency_ms": round(dur_ocr, 2),
        "asr_latency_ms": round(dur_asr, 2)
    }, f, indent=2, ensure_ascii=False)

print("=" * 70, flush=True)
print(f"🎉 HOÀN THÀNH ĐO ĐẠC HIỆU NĂNG AI! Kết quả lưu tại: {bench_file}", flush=True)
print("=" * 70, flush=True)
