# Video Retrieval System V3

Hệ thống truy xuất keyframe video bằng OpenCLIP, SQLite FTS (OCR/ASR) và trợ lý tìm kiếm trực quan chạy qua Antigravity CLI (`agy`). Giao diện web hỗ trợ tìm kiếm CLIP/OCR/ASR, xem preview video, filmstrip và chatbot kiểm chứng ứng viên bằng ảnh.

## 1. Yêu cầu

- Windows 10/11.
- Python 3.10–3.12 (khuyến nghị 3.11).
- Git và Antigravity CLI; lệnh `agy` phải chạy được trong terminal.
- Khoảng 2 GB dung lượng trống cho database, vectors, model và dependencies.

Kiểm tra trước khi cài:

```powershell
python --version
agy --help
```

Nếu `agy` đã cài nhưng không nằm trên `PATH`, khai báo `AGY_PATH` trong `.env` bằng đường dẫn tuyệt đối tới `agy.exe`.

## 2. Clone và cài dependency

```powershell
git clone https://github.com/PHamHuy-23/vision.git
cd vision
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

`requirements.txt` chứa cả các dependency thường bị thiếu trên máy mới: MCP SDK, HTTPX, CTranslate2, Transformers, SentencePiece, Sacremoses và multipart support.

Nếu PyTorch không phù hợp GPU/CUDA của máy, cài bản PyTorch phù hợp cho máy đó trước, sau đó chạy lại `pip install -r requirements.txt`. Hệ thống vẫn chạy bằng CPU nhưng sẽ chậm hơn.

## 3. Dữ liệu bắt buộc

Các file lớn không được lưu trên GitHub. Lấy chúng từ Drive/kho dữ liệu của nhóm và đặt ở thư mục gốc:

```text
vision/
├── all_vectors.npy          # ma trận OpenCLIP, 177321 vectors
├── video_index_v2.db        # metadata + OCR/ASR FTS
├── video_drive_metadata.json
├── video_fps_map.json
└── models/
    └── opus-mt-vi-en-ct2/   # tùy chọn; dịch offline
```

Nếu không có model dịch offline, hệ thống fallback sang `deep-translator` và cần Internet. Có thể đặt file dữ liệu ở nơi khác bằng `DB_PATH`, `CONSOLIDATED_VECTORS_PATH` và `DATA_ROOT` trong `.env`.

## 4. Cấu hình MCP cho Antigravity

Chạy tại thư mục project, thay đường dẫn bằng đường dẫn clone thực tế:

```powershell
agy mcp add video-researcher python "C:\duong-dan\toi\vision\mcp_server.py"
agy mcp list
```

Kết quả phải có `video-researcher`, loại `stdio`, trạng thái `enabled`. Nếu đã tồn tại, `mcp add` sẽ cập nhật cấu hình. Sau khi thay đổi `mcp_server.py`, hãy đóng session Agy cũ và restart backend để tránh MCP cũ còn nằm trong bộ nhớ.

## 5. Chạy server

```powershell
python -m src.main
```

Mở [http://localhost:8000](http://localhost:8000). Backend nhận request ngay, nhưng chatbot nhanh và ổn định nhất sau khi terminal hiện:

```text
[Startup] Flash model ready!
[Startup] Pro model ready!
```

Kiểm tra API:

```powershell
Invoke-RestMethod http://localhost:8000/api/v1/health
```

## 6. Chế độ tìm kiếm

- `Semantic`: mô tả hình ảnh/hành động; tiếng Việt được dịch sang tiếng Anh trước khi encode.
- `OCR`: chữ xuất hiện trên khung hình.
- `ASR`: lời thoại/phụ đề gắn với keyframe.
- Chatbot: tự phân rã truy vấn, tìm ứng viên, xem contact sheet và chỉ dùng sequence khi cần xác minh trình tự.

Thời gian chatbot phụ thuộc tải máy và dịch vụ Agy; trên máy phát triển hiện tại thường khoảng 30–50 giây sau prewarm. Đây không phải luồng tìm kiếm tức thời 5–10 giây.

## 7. Lỗi thường gặp

### `ModuleNotFoundError`

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

Hãy chắc chắn terminal đang dùng Python trong `.venv` (`Get-Command python`).

### `agy` không được nhận diện

Thêm thư mục chứa `agy.exe` vào `PATH`, hoặc khai báo `AGY_PATH` trong `.env`.

### Chatbot không thấy tool

Chạy `agy mcp list`; tên server bắt buộc là `video-researcher`. Cập nhật lại bằng lệnh ở mục 4 và restart Agy/backend.

### Thiếu database hoặc vectors

Kiểm tra chính xác hai tên `video_index_v2.db` và `all_vectors.npy`, hoặc đặt đường dẫn tuyệt đối trong `.env`.

### Dịch tiếng Việt chậm

Đặt model CTranslate2 tại `models/opus-mt-vi-en-ct2`. Nếu không có, hệ thống dùng dịch vụ dịch online.

## 8. Cấu trúc runtime

```text
frontend/index.html       giao diện web
mcp_server.py             các tool Agy gọi
src/main.py               FastAPI và API endpoints
src/sqlite_engine.py      OpenCLIP/FAISS/SQLite retrieval
src/agy_session.py        lifecycle, routing và prompt của Agy
src/fast_translator.py    dịch Việt–Anh và cache
src/supabase_service.py   metadata Supabase tùy chọn
```
