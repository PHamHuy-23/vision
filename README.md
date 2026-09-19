# Video Retrieval System V3 (AI Challenge)

Đây là hệ thống truy xuất video nâng cao (Video Retrieval System V3) được thiết kế chuyên biệt cho AI Challenge. Hệ thống sử dụng kiến trúc tìm kiếm đa phương thức kết hợp giữa **OpenCLIP**, **SQLite/FAISS**, hỗ trợ tìm kiếm bằng ngôn ngữ tự nhiên, OCR, ASR, Temporal Search, và đặc biệt tích hợp **Antigravity CLI (Multi-Agent AI)**.

---

## 🌟 1. Các Tính Năng Nổi Bật (V3)
- **AI Agentic Retrieval (Antigravity):**
  - **Tích hợp Gemini Flash/Pro:** Hệ thống tự động phân luồng (Model Router) dựa trên độ phức tạp của câu hỏi.
  - **Pre-warming & Parallel Tools:** AI được nạp sẵn vào RAM, thực thi song song các công cụ (Semantic, OCR, ASR) giúp giảm thời gian phản hồi từ 150s xuống <10s.
  - **Auto-Translation:** Tự động dịch tiếng Việt sang tiếng Anh qua `deep_translator` hoặc `MyMemoryTranslator` trước khi đưa vào OpenCLIP.
- **Đa phương thức tìm kiếm:**
  - **Semantic AI (CLIP):** Tìm kiếm theo ngữ nghĩa (ViT-B-32).
  - **Hierarchical Search (Mới):** Cấu trúc 2 tầng (Coarse-to-Fine) Max-Pooling cho phép tìm kiếm qua 177.000 vectors với tốc độ 0.05s.
  - **OCR & ASR Exact Match:** Hỗ trợ tìm kiếm theo text trên khung hình hoặc lời thoại.
  - **Temporal Search:** Tìm kiếm theo chuỗi sự kiện `A -> B`.
- **Hạ tầng Ảnh & Video:**
  - **Cloudflare R2 Integration:** Các khung hình được tự động nạp từ R2 (`.webp`) để tối ưu tốc độ load. Hệ thống có cơ chế "Bảo hiểm kép" (Fallback) tự động gọi Google Drive nếu R2 bị sập qua thẻ `onerror`.
- **Giao diện (ChatGPT Style):** 
  - UI hiện đại với đồng hồ bấm giờ (Live timer), huy hiệu quá trình suy nghĩ của AI, và chế độ gập gọn tool sau khi hoàn thành.
  - Cửa sổ Preview Modal lớn (`75vh`) tích hợp nhúng (Iframe) tự tua video.

---

## 📁 2. Chuẩn Bị Dữ Liệu & Antigravity (Bắt Buộc)

### A. Dữ liệu cốt lõi
Bạn cần tải các file dữ liệu từ Drive của team và đặt vào thư mục gốc của project (cùng cấp thư mục `src`):
1. `all_vectors.npy`: File ma trận vector (177,321 vectors).
2. `video_index.db`: Database SQLite chứa thông tin metadata, OCR, ASR, Object, GDrive File ID.
3. `video_fps_map.json`: File map Frame Per Second.
4. `.env`: File chứa các cấu hình API Key.

### B. Cấu hình Antigravity CLI & MCP Server
Hệ thống sử dụng Antigravity CLI để kích hoạt AI Agent. Bất kỳ ai clone dự án về **bắt buộc** phải cài đặt và cấu hình MCP Tools:

1. **Cài đặt Antigravity CLI:** Yêu cầu máy cài đặt sẵn công cụ Antigravity CLI (`agy`) để có thể chạy AI Agent.
2. **Cấu hình MCP Video Researcher (Rất quan trọng):**
   Trong mã nguồn có sẵn một file tên là `mcp_server.py`. Đây chính là cầu nối giữa AI và Database.
   Bạn cần mở file cấu hình của Antigravity (`~/.gemini/config/mcp_config.json`) và thêm cấu hình trỏ tới file này như sau:
   ```json
   "mcpServers": {
     "video-researcher": {
       "command": "python",
       "args": ["<ĐƯỜNG_DẪN_TUYỆT_ĐỐI_TỚI_THƯ_MỤC_CLONE>/mcp_server.py"]
     }
   }
   ```
3. **Bộ não AI (System Prompt) nằm ở đâu?:**
   Toàn bộ "Tính cách", chỉ thị ép gọi Parallel Tools, ép dịch tiếng Anh (Auto-translate) và định dạng xuất HTML đều được lập trình cứng (Hardcode) bên trong file **`src/agy_session.py`** (Tại khối lệnh `if self.is_first_message:`). Bạn có thể vào đó để "dạy" thêm cho AI nếu muốn!

---

## 🚀 3. Hướng Dẫn Cài Đặt & Chạy Local

### Bước 1: Cài đặt Python
Yêu cầu máy tính cài đặt **Python 3.10** trở lên. Khuyến nghị sử dụng môi trường ảo (Virtual Environment) hoặc Conda.

### Bước 2: Cài đặt thư viện
Mở Terminal / CMD tại thư mục chứa project và chạy lệnh:
```bash
pip install -r requirements.txt
pip install deep-translator  # Đã được thêm vào để hỗ trợ tự động dịch
```

### Bước 3: Khởi động Backend Server
Chạy lệnh sau để khởi động hệ thống Backend:
```bash
python -m src.main
```
*(Quá trình khởi động sẽ mất khoảng 5-10 giây do Server cần load mô hình OpenCLIP (PyTorch) vào RAM/GPU và Pre-warm các session AI).*

### Bước 4: Truy cập Giao Diện
Mở trình duyệt (Chrome/Edge/Firefox) và truy cập vào địa chỉ:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## ⚡ 4. Hướng Dẫn Tối Ưu Tốc Độ

Hệ thống cung cấp tốc độ phản hồi khác nhau tùy thuộc vào cách bạn tương tác:
1. **Tìm kiếm siêu tốc (0.05s):** Ở ô Manual Text Search, hãy nhập **tiếng Anh** (Ví dụ: `black dog`). Hệ thống sẽ bỏ qua bước dịch thuật và truy xuất database ngay lập tức.
2. **Tìm kiếm tiếng Việt (~0.8s):** Khi bạn nhập tiếng Việt, Backend sẽ tự động gọi Google Translate (có fallback sang MyMemory) để dịch sang tiếng Anh, tốn thêm khoảng 0.5s.
3. **Tìm kiếm AI (Trợ lý Agent):** Khi sử dụng thanh chat AI, Agent sẽ kích hoạt Gemini Flash/Pro, tự động dịch và gọi song song các Tool, tổng thời gian trả lời khoảng 5-10s.

---

## 🛠️ 5. Hướng Dẫn Cài Đặt Tool "Nhảy Cóc Video Google Drive"
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

Bây giờ bạn chỉ việc tìm kiếm, phân tích và chọn các frame chính xác để Submission! Chúc team đạt kết quả tốt nhất! 🏆
