# Video Retrieval System (AI Challenge)

Đây là hệ thống truy xuất video nâng cao (Video Retrieval System) sử dụng mô hình AI **OpenCLIP**, cho phép tìm kiếm qua ngôn ngữ tự nhiên (hỗ trợ Tiếng Việt) và tìm kiếm ảnh tương tự (Vector Similarity Search) cực nhanh bằng **FAISS**.

Hệ thống được thiết kế tối ưu với giao diện Web (TailwindCSS) đi kèm công cụ nén file nộp bài chuẩn định dạng KIS, QA, TRAKE.

---

## 1. Kiến trúc hệ thống
- **Backend:** FastAPI, Python, PyTorch, FAISS.
- **Frontend:** HTML, JavaScript thuần (Vanilla JS), TailwindCSS.
- **Tính năng cốt lõi:**
  - Tự động dịch Query Tiếng Việt -> Tiếng Anh (sử dụng `deep-translator`).
  - Tách từ khóa (Split keywords) và trung bình hóa Vector (Compositional Vector Averaging) để tìm kiếm nhiều chi tiết cùng lúc mà không cần AI phức tạp.
  - Sidebar đóng gói Submission chuẩn KIS, QA, TRAKE.
  - Tích hợp Iframe Google Drive nhảy cóc đúng giây (Yêu cầu cài Tampermonkey).

---

## 2. Chuẩn bị Dữ liệu (Rất Quan Trọng)
Mỗi thành viên trong team sau khi Clone Code về cần tải các file Data từ Google Drive chung của Team và đặt vào thư mục gốc của project:
https://drive.google.com/file/d/1oiLUeSKdPydvMrntE5d6W3FUSbDJ93eO/view?usp=sharing
1. `all_vectors.npy` (~350MB): File chứa toàn bộ vector đã được encode bằng CLIP.
2. `frame_map_supabase.json` (~177MB): File map metadata (đường dẫn, thời gian, frame_idx) của toàn bộ frame.
3. `video_drive_metadata.json`: File chứa Google Drive ID của các video để bật Iframe.
4. `video_index.db` (nếu có): Database sao lưu cục bộ (SQLite).
5. .env()

*Lưu ý: Chỉ cần Code và Data nằm chung 1 thư mục là hệ thống đã sẵn sàng.*

---

## 3. Hướng dẫn Cài đặt & Chạy Local

### Bước 1: Cài đặt Python
Đảm bảo máy đã cài **Python 3.10** trở lên. (Khuyên dùng môi trường ảo `venv` hoặc `Conda`).

### Bước 2: Cài đặt thư viện
Mở Terminal/CMD tại thư mục project và chạy lệnh:
```bash
pip install -r requirements.txt
```

### Bước 3: Khởi động Backend Server
Chạy lệnh sau để bật Server :
```bash
python -m src.main
```
*Server sẽ chạy ở địa chỉ: `http://localhost:8000`.*

### Bước 4: Mở giao diện Web
Bạn chỉ cần mở trình duyệt (Chrome/Edge) và truy cập:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 4. Hướng dẫn cài Tool "Nhảy Cóc Video Google Drive"
Để tính năng **Play Video** nhảy được tới đúng khung thời gian (Timestamp), thành viên cần cài một đoạn Script nhỏ vào trình duyệt:

1. Cài Extension **Tampermonkey** cho trình duyệt (tìm trên Chrome Web Store).
2. Bấm biểu tượng Tampermonkey -> **Create a new script**.
3. Copy toàn bộ đoạn code dưới đây dán vào và ấn **Ctrl + S** để lưu lại:

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

Giờ thì bạn chỉ cần tìm kiếm, ném vào rổ Submission và chiến đấu thôi! Chúc team đạt kết quả cao! 🚀
