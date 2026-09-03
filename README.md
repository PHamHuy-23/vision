# Video Retrieval System V3 (AI Challenge)

Đây là hệ thống truy xuất video nâng cao (Video Retrieval System V3) được thiết kế chuyên biệt cho AI Challenge. Hệ thống sử dụng kiến trúc tìm kiếm đa phương thức kết hợp giữa **OpenCLIP**, **FAISS** và **BM25**, hỗ trợ tìm kiếm bằng ngôn ngữ tự nhiên (hỗ trợ Tiếng Việt), tìm kiếm bằng hình ảnh, OCR, ASR, và tìm kiếm chuỗi thời gian (Temporal Search).

Hệ thống cung cấp một Backend mạnh mẽ bằng FastAPI, giao diện Frontend tối ưu bằng TailwindCSS, tích hợp Google Drive Proxy, và công cụ đóng gói file nộp bài chuẩn KIS, QA, TRAKE.

---

## 🚀 1. Các Tính Năng Nổi Bật (V3)
- **Đa phương thức tìm kiếm (Multi-Modal Search):**
  - **Semantic AI (CLIP):** Tự động dịch Tiếng Việt -> Tiếng Anh, phân tách từ khóa (Split keywords) để tìm kiếm gộp (Compositional Vector Averaging).
  - **OCR & ASR Exact Match:** Hỗ trợ tìm kiếm theo text xuất hiện trong khung hình hoặc lời thoại (với fuzzy matching và từ đồng nghĩa).
  - **Temporal Search:** Tìm kiếm theo chuỗi sự kiện (Ví dụ: `xe dừng lại -> người bước ra`) trong khoảng thời gian nhất định.
  - **Image Search:** Tìm kiếm các khung hình tương đồng dựa trên một hình ảnh tải lên.
- **Tích hợp Google Drive & Supabase:** Tự động proxy ảnh từ Google Drive để tránh lỗi 403 Rate Limits, sử dụng cache memory LRU cho tốc độ tải cực nhanh. Fallback meta data thông qua Supabase/SQLite.
- **Submission Builder:** Sidebar tích hợp sẵn cho phép người dùng chọn các frame và tự động build ra file zip chuẩn format nộp bài (KIS, QA, TRAKE) của cuộc thi.

---

## 📂 2. Chuẩn Bị Dữ Liệu (Bắt Buộc)
Để hệ thống hoạt động, mã nguồn cần đi kèm với các file dữ liệu (Data). Các thành viên trong team sau khi clone code cần tải các file sau từ Google Drive chung và đặt vào thư mục gốc của project:
*(Link dữ liệu nội bộ của team: [https://drive.google.com/file/d/1P7RJOPmtizVqmXcKSnhlhfMePM050nIv/view?usp=sharing])*

1. `all_vectors.npy`: File ma trận vector FAISS (cực kỳ quan trọng để search CLIP).
2. `frame_map_supabase.json`: File map metadata chứa thông tin OCR, ASR, Object, và Google Drive File ID của toàn bộ các khung hình.
3. `video_drive_metadata.json`: File chứa Google Drive ID của các video để hiển thị Iframe phát video.
4. `video_fps_map.json`: File ánh xạ Frame Per Second để tính toán chính xác frame_idx từ thời gian thực.
5. `video_index.db` / `video_index_v2.db`: Database cục bộ (SQLite) phục vụ fallback dữ liệu.
6. `.env`: File chứa các cấu hình API Key (Supabase, Google Drive, v.v.).

*Lưu ý: Đảm bảo các file dữ liệu này nằm cùng cấp với thư mục `src` và `frontend`.*

---

## ⚙️ 3. Hướng Dẫn Cài Đặt & Chạy Local

### Bước 1: Cài đặt Python
Yêu cầu máy tính cài đặt **Python 3.10** trở lên. Khuyến nghị sử dụng môi trường ảo (Virtual Environment) hoặc Conda để tránh xung đột thư viện.

### Bước 2: Cài đặt thư viện
Mở Terminal / CMD tại thư mục chứa project (nơi có file `requirements.txt`) và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi động Backend Server
Chạy lệnh sau để khởi động hệ thống Backend:
```bash
python -m src.main
```
*(Nếu muốn server tự động reload khi sửa code, bạn có thể dùng lệnh: `uvicorn src.main:app --reload`)*

### Bước 4: Truy cập Giao Diện
Mở trình duyệt (Chrome/Edge/Firefox) và truy cập vào địa chỉ:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 🛠 4. Hướng Dẫn Cài Đặt Tool "Nhảy Cóc Video Google Drive"
Để tính năng **Play Video** nhảy được tới đúng khung thời gian (Timestamp) ngay trong Iframe, thành viên cần cài một đoạn Script nhỏ vào trình duyệt:

1. Cài đặt Extension **Tampermonkey** cho trình duyệt (Tìm trên Chrome Web Store).
2. Bấm vào biểu tượng Tampermonkey -> Chọn **Create a new script**.
3. Xóa code mặc định, copy đoạn code dưới đây dán vào và nhấn **Ctrl + S** để lưu lại:

```javascript
// ==UserScript==
// @name         Google Drive Video Timestamp Jumper
// @namespace    http://tampermonkey.net/
// @version      1.0
// @description  Ép Google Drive tự động tua tới đúng giây trong Iframe
// @match        https://drive.google.com/file/d/*/preview*
// @grant        none
// ==/UserScript==

(function() {
    'use strict';
    const hash = window.location.hash;
    if (hash.startsWith('#t=')) {
        const timeToSeek = parseFloat(hash.replace('#t=', ''));
        if (!isNaN(timeToSeek)) {
            const checkVideo = setInterval(() => {
                const video = document.querySelector('video');
                if (video && video.readyState >= 1) {
                    video.currentTime = timeToSeek;
                    video.play();
                    clearInterval(checkVideo);
                }
            }, 500);
        }
    }
})();
```

Bây giờ bạn chỉ việc tìm kiếm, phân tích và chọn các frame chính xác để Submission! Chúc team đạt kết quả tốt nhất! 🚀
