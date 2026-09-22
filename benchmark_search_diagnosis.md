# Báo cáo chẩn đoán độ trễ OCR / ASR / CLIP

Ngày kiểm thử: 2026-09-21

## Kết luận

SQLite FTS5, metadata cache và MCP wrapper không phải nút thắt của OCR/ASR.
Ở đường Search trực tiếp, OCR và ASR nhanh hơn CLIP. Hai vấn đề thực tế được
phát hiện là:

1. Endpoint thử nghiệm `/api/v1/search/all` từng chạy CLIP, OCR và ASR đồng
   thời, gây tranh chấp CPU/thread. Endpoint này không được frontend/MCP sử
   dụng và đã được xoá sau kiểm thử.
2. Telemetry giao diện chờ toàn bộ thumbnail, trong khi ảnh thứ 7 trở đi dùng
   lazy-load. Bộ đếm vì thế có thể đứng mãi ở `28/50` nếu người dùng không
   cuộn trang, tạo cảm giác truy vấn vẫn còn chạy.

Ngoài ra, đường AI Assistant/Agy hiện trả `Model Router` rồi `[DONE]` mà không
thấy MCP tool thực sự được gọi. Đây là lỗi correctness/lifecycle riêng, nên
không thể dùng thời gian AI Assistant hiện tại để so sánh tốc độ retrieval.

## Giai đoạn 1 — Telemetry

Đã thêm các chỉ số:

- `db_connect_ms`
- `fts_execute_ms`
- `fts_fetch_ms`
- `metadata_ms`
- cache hit và DB fallback
- FTS/fuzzy fallback
- serialize và kích thước response
- browser header/body parse/DOM render
- ảnh đầu tiên và 6 ảnh eager đầu tiên

Mỗi request có `request_id` trong payload và log backend.

## Giai đoạn 2 — So sánh theo tầng

Đo 30 lần/query, warm-up 3 lần, `top_k=20`.

| Mode | Query | FTS median | Backend median | HTTP median | HTTP P95 |
|---|---|---:|---:|---:|---:|
| OCR | HANOI | 2,84 ms | 3,30 ms | 14,24 ms | 52,57 ms |
| OCR | camera | 3,02 ms | 2,84 ms | 10,68 ms | 43,19 ms |
| ASR | xin chào | 3,91 ms | 4,37 ms | 13,10 ms | 48,47 ms |
| ASR | hello | 2,33 ms | 2,67 ms | 13,96 ms | 40,75 ms |

- Fuzzy fallback: 0 lần.
- Metadata cache: được nạp, cache hit 100%.
- Metadata hydration median: dưới 0,05 ms.

## Giai đoạn 3 — Top K và cạnh tranh tài nguyên

Với `top_k` 10, 20, 50 và 100, median HTTP của OCR/ASR nằm trong khoảng
7–27 ms. Tăng `top_k` không giải thích được mức chậm x10.

Khi gọi `/api/v1/search/all` để chạy đồng thời với CLIP:

| Query | CLIP trung bình | OCR trung bình | ASR trung bình | Spike text lớn nhất |
|---|---:|---:|---:|---:|
| camera | 145,8 ms | 10,9 ms | 51,2 ms | ASR 487 ms |
| HANOI | 1.105,9 ms | 59,7 ms | 58,0 ms | OCR 651 / ASR 645 ms |
| hello | 140,6 ms | 32,6 ms | 4,9 ms | OCR 333 ms |
| xin chào | 150,1 ms | 106,4 ms | 12,0 ms | OCR 1.182 ms |

## Giai đoạn 4 — Trình duyệt thực tế

Đo tại giao diện localhost, `top_k=50`:

| Mode | Query | API | Ảnh đầu |
|---|---|---:|---:|
| OCR | camera | 30 ms | 212 ms |
| ASR | xin chào | 46 ms | 184 ms |
| CLIP | a camera | 201 ms | 336 ms |

Sau khi sửa telemetry lazy-load, lượt xác nhận ASR cho kết quả:

- API: 46 ms
- Ảnh đầu: 186 ms
- 6 ảnh eager đầu: 206 ms
- Backend search: 9,63 ms
- Parse JSON: 2 ms
- Render DOM: 14 ms

## Giai đoạn 5 — MCP và AI Assistant

MCP wrapper, 20 lần/mode:

| Tool | Median | Max |
|---|---:|---:|
| CLIP | 190,0 ms | 915,3 ms |
| OCR | 36,8 ms | 87,7 ms |
| ASR | 39,6 ms | 83,0 ms |

MCP wrapper không tạo mức chậm x10 cho OCR/ASR. Tuy nhiên `/api/v1/chat` chỉ
phát hai SSE event (`Model Router`, `[DONE]`) và không phát event tool thực tế.
Cần sửa lifecycle của session Agy trước khi benchmark AI Assistant.

## Thứ tự xử lý đề xuất

1. Đã sửa Agy session để phát hiện process chết, stdin đóng hoặc stdout EOF;
   session hỏng được huỷ và yêu cầu tiếp theo sẽ khởi tạo lại.
2. Đã xoá `/api/v1/search/all` cùng request model/helper không được sử dụng.
3. Đã giữ telemetry 6 ảnh eager; không dùng toàn bộ ảnh lazy-load làm điều kiện
   kết thúc.
4. Cần chạy lại benchmark chat với Agy thật sau khi backend được khởi động lại.
