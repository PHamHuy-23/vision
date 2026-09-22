# Benchmark tốc độ search OCR / ASR

Ngày đo: 2026-09-21  
Cách đo: gọi trực tiếp SQLite FTS5 (`ocr_fts`, `asr_fts`), `top_k=20`, warm-up 3 lần, đo 40 lần/query trên máy local.

## Kết quả

| Mode | Query | FTS rows | Hits | Mean | Median | P95 | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| OCR | HANOI | 69,160 | 9 | 2.043 ms | 1.752 ms | 4.913 ms | 5.226 ms |
| OCR | camera | 69,160 | 20 | 0.942 ms | 0.853 ms | 1.530 ms | 3.375 ms |
| ASR | xin chào | 159,256 | 20 | 1.472 ms | 1.451 ms | 2.016 ms | 3.429 ms |
| ASR | hello | 159,256 | 20 | 0.327 ms | 0.279 ms | 0.541 ms | 0.860 ms |

## Nhận xét

- OCR trung bình: **1.49 ms**, ASR trung bình: **0.90 ms** trên bộ query mẫu.
- P95 cao nhất là OCR `HANOI`: **4.913 ms**; chưa thấy dấu hiệu FTS5 là nút thắt.
- Benchmark này chưa tính auto-translate, khởi động process/model, serialize JSON, network API, tải thumbnail hoặc render UI.
- Vì vậy cần so sánh với benchmark cũ theo đúng tầng: benchmark cũ đo end-to-end và có các truy vấn OCR/ASR lần lượt khoảng **1,010 ms** và **1,642 ms** backend; phần chênh lệch nằm ngoài truy vấn FTS trực tiếp.
