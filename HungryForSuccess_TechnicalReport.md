# HỘI THI AI CHALLENGE THÀNH PHỐ HỒ CHÍ MINH NĂM 2026
## BÁO CÁO KỸ THUẬT: HỆ THỐNG TRUY XUẤT VIDEO (VIDEO RETRIEVAL SYSTEM V3)
**KIẾN TRÚC TỐI ƯU CỤC BỘ LOCAL-FIRST KẾT HỢP OPENCLIP, FAISS, SQLITE VÀ HẠ TẦNG GOOGLE DRIVE CLOUD**

---

### THÔNG TIN ĐỘI THI
- **Tên đội thi:** `HungryForSuccess(๑>•̀๑)`
- **Tên định dạng nộp danh sách:** `[Mã số] - [Bảng thi] - HungryForSuccess(๑>•̀๑)`
- **Họ và tên trưởng nhóm/đại diện:** Phạm Huy
- **Email liên hệ:** `huy593880@gmail.com`
- **Giai đoạn tham gia:** Vòng Sơ tuyển (21/8/2026, 28/8/2026 và 04/9/2026)

---

### TÓM TẮT GIẢI PHÁP (ABSTRACT - 176 TỪ)
Hệ thống Video Retrieval System V3 của đội HungryForSuccess(๑>•̀๑) tham dự AI Challenge TP.HCM 2026 được xây dựng theo kiến trúc Local-First hiệu năng cao kết hợp lưu trữ đám mây Google Drive, hoàn toàn không phụ thuộc vào các mô hình Generative AI hay API bên thứ ba. Hệ thống quản lý toàn diện 873 video với 177.321 keyframes, sử dụng OpenCLIP (ViT-B/32) trích xuất sẵn vector đặc trưng 512 chiều (`all_vectors.npy` ~346MB) và thư viện FAISS IndexFlatIP cho tốc độ truy xuất tương đồng vector cực nhanh dưới 15ms. Toàn bộ thông tin đối tượng (174.585 frames), giọng nói ASR (159.256 frames), ký tự OCR (69.160 frames) cùng tọa độ Bounding Box được lưu trữ trong SQLite cục bộ (`video_index_v2.db` ~814MB). Đội thi phát triển các giải pháp kỹ thuật chuyên sâu: bộ đệm Reverse Proxy LRU Cache và Semaphore(8) giải quyết triệt để lỗi chặn 403 của Google Drive, Userscript Tampermonkey tự động nhảy cóc video iframe theo timestamp, công cụ quy đổi thời gian sang frame_idx bằng FPS map, truy vấn chuỗi thời gian (Temporal Search), duyệt ngữ cảnh lân cận (Frame Context Search), và Submission Builder tự động đóng gói bài nộp KIS, QA, TRAKE chuẩn định dạng zip.

---

## 1. GIẢI PHÁP, THUẬT TOÁN VÀ CHIẾN LƯỢC HỆ THỐNG

### 1.1. Kiến trúc Tổng thể và Triết lý Thiết kế (Architecture & Core Philosophy)
Hệ thống Video Retrieval System V3 được phát triển dựa trên nguyên tắc vận hành tất định (Deterministic), tập trung tối đa vào tốc độ phản hồi tính bằng mili-giây, độ ổn định tuyệt đối và khả năng hoạt động độc lập không phụ thuộc vào các mô hình trí tuệ nhân tạo sinh (Generative AI) hay các dịch vụ đám mây bên ngoài vốn tiềm ẩn nhiều rủi ro trễ mạng, chạm giới hạn tần suất (Rate Limit) hoặc mất kết nối trong phòng thi đấu thực tế. Hệ thống gồm 5 tầng phân cấp:

| Phân tầng Hệ thống | Thành phần Kỹ thuật & Dữ liệu Thực tế Vận hành |
| :--- | :--- |
| **1. Google Drive Cloud Storage** | Lưu trữ 873 video gốc và 177.321 keyframe hình ảnh trên đám mây Google Drive (hơn 519.519 tệp/thư mục). Ánh xạ toàn bộ đường dẫn vật lý thành Google Drive File ID và CDN URL. |
| **2. Vector Feature Store** | Trích xuất đặc trưng hình ảnh bằng OpenCLIP ViT-B/32 thành ma trận vector toàn cục `all_vectors.npy` (177.321 vector x 512 chiều, float32, dung lượng 346.3 MB). Nạp vào bộ nhớ bằng kỹ thuật Memory-Mapping (`mmap_mode='r'`). |
| **3. Local Metadata Database (SQLite)** | Toàn bộ dữ liệu metadata được nạp vào SQLite cục bộ `video_index_v2.db` (813.9 MB) gồm 177.321 bản ghi: 174.585 frames có nhãn Object, 159.256 frames có ASR, 69.160 frames có OCR, lưu đầy đủ tọa độ Bounding Box. |
| **4. High-Speed Retrieval Engine** | Sử dụng FAISS (IndexFlatIP) tính tích vô hướng cực nhanh (10-15ms). Bộ máy tìm kiếm mờ (Fuzzy Text Search) trên nhãn đối tượng, nội dung OCR và giọng nói ASR bằng bảng chỉ mục đảo ngược (Inverted Index). |
| **5. Interactive UI & Submission Tool** | Giao diện Web SPA TailwindCSS gồm 5 Tab: Text Search, Image Search, Video Interval, Time ➔ Frame, Frame Context. Trình Submission Builder tự động đóng gói bài nộp KIS, QA, TRAKE chuẩn zip. |

---

### 1.2. Kiến trúc Lưu trữ Google Drive và Các Giải pháp Kỹ thuật Chuyên sâu
Một trong những điểm nhấn kỹ thuật quan trọng nhất của đội là việc giải quyết bài toán lưu trữ và phân phối tập dữ liệu video dung lượng khổng lồ trên hạ tầng đám mây Google Drive phục vụ thi đấu trực tiếp:

1. **Thách thức 1: Giới hạn tần suất và nguy cơ bị chặn truy cập (HTTP 403 Rate Limits của Google Drive):**
   - *Thực trạng:* Khi thực hiện tìm kiếm Top 50 hoặc Top 100 kết quả, trình duyệt gửi đồng loạt 50-100 yêu cầu tải ảnh về Google Drive, dễ bị kích hoạt cơ chế bảo vệ chống DDoS trả về mã lỗi 403 Forbidden.
   - *Giải pháp:* Đội xây dựng bộ phận Reverse Proxy bất đồng bộ tại máy chủ FastAPI (`/api/v1/drive/proxy/{file_id}`) kết hợp `asyncio.Semaphore(8)` khống chế tối đa 8 luồng kết nối đồng thời và bộ nhớ đệm `functools.lru_cache(maxsize=1000)`. Đồng thời ưu tiên Google Drive Thumbnail API (`drive.google.com/thumbnail?id={file_id}&sz=w400`) có trần chịu tải cao hơn nhiều so với liên kết tải trực tiếp.

2. **Thách thức 2: Xác thực và kiểm chứng video dung lượng lớn trong phòng thi:**
   - *Thực trạng:* Tải toàn bộ file video gốc (hàng trăm MB đến cả GB) về máy tính cá nhân là hoàn toàn bất khả thi trong khung thời gian vài phút của một câu hỏi.
   - *Giải pháp:* Tích hợp trình phát Iframe trực tiếp từ Google Drive (`drive.google.com/file/d/{file_id}/preview#t={pts_time}`). Đặc biệt, đội tự lập trình đoạn mã UserScript Tampermonkey **"Google Drive Video Timestamp Jumper"** tự động bắt hash `#t=time` và điều khiển thẻ `<video>` bên trong iframe tua tức thời đến đúng giây thực tế và phát ngay, cho phép thí sinh đối soát 5s trước và sau mà không tốn băng thông tải video.

3. **Thách thức 3: Sai lệch giữa Mốc thời gian thực tế và Chỉ số khung hình (Timestamp vs Frame Index):**
   - *Thực trạng:* Cuộc thi yêu cầu nộp `frame_idx` thực tế chứ không chấp nhận `frame_number`. Các video có FPS khác nhau (từ 24.0 đến 30.0 fps), tính công thức mặc định sẽ sai lệch hàng chục frame.
   - *Giải pháp:* Trích xuất chỉ số FPS chính xác của toàn bộ 873 video vào `video_fps_map.json`. Công cụ "Time ➔ Frame Converter" trên giao diện tự tra cứu FPS và tính toán chính xác: $\text{frame\_idx} = \text{round}(\text{time\_sec} \times \text{actual\_fps})$.

4. **Thách thức 4: Nguy cơ nghẽn mạng phòng thi đối với cơ sở dữ liệu:**
   - *Thực trạng:* Truy vấn metadata qua mạng từ cơ sở dữ liệu đám mây (như Supabase) có độ trễ 300ms - 2000ms và dễ gián đoạn nếu mất mạng.
   - *Giải pháp:* Chuyển toàn bộ 177.321 bản ghi metadata sang SQLite cục bộ (`video_index_v2.db` 813.9 MB), thiết lập chỉ mục `idx_video_id` và `idx_pts_time`. Mọi thao tác tra cứu diễn ra Offline 100% với độ trễ < 5ms.

5. **Thách thức 5: Nguy cơ vi phạm quy chế định dạng bài nộp (Human Error):**
   - *Thực trạng:* Sao chép thủ công dễ nhầm dấu phẩy, sót dòng, quên xóa đuôi `.mp4`, hoặc quá 100 dòng.
   - *Giải pháp:* Submission Builder tự động hóa 100%, kiểm soát tối đa 100 dòng, tự loại bỏ `.mp4`, phân loại KIS, QA, TRAKE và nén `submission.zip` chứa đúng thư mục `submission/` theo hướng dẫn kỹ thuật BTC.

---

### 1.3. Các Phương pháp và Thuật toán Tìm kiếm Đa phương thức Thực tế
- **Semantic Vector Search (OpenCLIP + FAISS IndexFlatIP):** Vector 512 chiều, chuẩn hóa L2, truy xuất quét cạn chính xác 100% trong 10-15ms.
- **Dual Fallback Translation:** Dịch tự động tiếng Việt sang tiếng Anh bằng `GoogleTranslator`, tự động chuyển tiếp `MyMemoryTranslator` khi có sự cố.
- **Compositional Vector Search (Toán tử `+`):** Nhập cú pháp `A + B` (ví dụ `red car + dog`), hệ thống mã hóa từng cụm từ riêng biệt, chuẩn hóa và tính vector trung bình $\vec{V}_{avg} = \frac{\text{mean}(\vec{v}_i)}{\|\text{mean}(\vec{v}_i)\|}$.
- **Exact & Fuzzy Matching cho OCR và ASR:** Áp dụng trên 69.160 frames OCR và 159.256 frames ASR, tích hợp từ điển đồng nghĩa (xe hơi $\leftrightarrow$ ô tô $\leftrightarrow$ car) và so khớp mờ Levenshtein.
- **Temporal Sequence Search (Chuỗi thời gian `->`):** Cú pháp `query1 -> query2`, truy xuất song song Top 200 ứng viên và ghép cặp trong cùng video thỏa mãn: $0 < frame_2 - frame_1 \le 900$ (trong vòng 30 giây).
- **Reverse Image Search (Ảnh mẫu qua Ctrl+V):** Dán ảnh từ Clipboard, sinh embedding qua `encode_image` và truy xuất FAISS tức thì.
- **Frame Context Search (Ngữ cảnh xung quanh frame mục tiêu):** Nhập `VideoID, FrameIdx`, cơ chế `surrounding=True` truy xuất các khung hình liền kề cả trước và sau theo khoảng cách tuyệt đối `ORDER BY ABS(frame_idx - ?) ASC LIMIT ?`.

---

## 2. GIAO DIỆN HỆ THỐNG VÀ CÔNG CỤ TƯƠNG TÁC
Giao diện người dùng (`frontend/index.html`) được xây dựng dạng Single Page Application (SPA) trên nền TailwindCSS, giao diện Dark Mode tối ưu cho thi đấu:

1. **Bảng điều khiển 5 Tab chức năng:**
   - **Tab 1 - Text Search:** Semantic AI (OpenCLIP), OCR (chữ trên hình), ASR (giọng nói), hỗ trợ toán tử `+` và `->`.
   - **Tab 2 - Image Search:** Kéo thả, duyệt file hoặc dán ảnh trực tiếp từ Clipboard bằng phím tắt **Ctrl+V**.
   - **Tab 3 - Video Interval:** Rà soát toàn bộ keyframes trong phân đoạn video từ Start Time đến End Time.
   - **Tab 4 - Time ➔ Frame Converter:** Nhập MM:SS, tra cứu FPS chuẩn và tính `frame_idx` tức thì.
   - **Tab 5 - Frame Context:** Nhập `VideoID, FrameIdx` (ví dụ `L21_V022, 38228`), hiển thị 20-50 khung hình bao quanh.
2. **Lưới Thẻ Kết quả (Results Grid):** Ảnh thumbnail chất lượng cao, Video ID, Frame Index, Timestamp (MM:SS), Điểm số tương đồng (% Score), nút "Similar", "Track" và nút "+" thêm vào bài nộp.
3. **Cửa sổ Thẩm định Khung hình (Inspection Modal):**
   - Phóng to ảnh, điều hướng phím mũi tên Trái/Phải hoặc vuốt chuột.
   - **Bounding Box Overlay:** Hộp đỏ thể hiện Object Detection, hộp xanh dương thể hiện OCR kèm nhãn và độ tin cậy.
   - **Play Video (Iframe):** Nhúng Google Drive video, kết hợp Tampermonkey tua tự động đến đúng giây `#t=pts_time`.
4. **Thanh Công cụ Đóng gói Bài nộp Tự động (Submission Builder Sidebar):**
   - Tự động lưu phiên vào `localStorage`.
   - Hỗ trợ KIS, QA, TRAKE (nút "+ Tách chuỗi mới").
   - Kiểm tra giới hạn 100 dòng, tự loại bỏ `.mp4`, xuất tệp `submission.zip` chuẩn quy cách BTC.

---

## 3. PHÂN TÍCH MỘT SỐ TÌNH HUỐNG TRUY VẤN TIÊU BIỂU

### 3.1. Tình huống 1: Truy vấn Ngữ nghĩa Đa thuộc tính (Đợt thi 21/08/2026)
- **Câu truy vấn:** *“Người đàn ông mặc áo sơ mi xanh dương đang bắt tay một người phụ nữ mặc áo trắng trong văn phòng”*
- **Đặc điểm & Độ khó:** Chứa nhiều thực thể (nam, nữ), màu áo đối lập (xanh dương, trắng), hành động (bắt tay) và bối cảnh (văn phòng). CLIP thông thường dễ bị hiện tượng trộn thuộc tính (gán nhầm màu áo).
- **Cách tiếp cận & Thuật toán:** Sử dụng Compositional Search với toán tử `+`: nhập `man in blue shirt + handshaking with woman in white + modern office`. Hệ thống trích xuất vector từng mệnh đề độc lập qua OpenCLIP ViT-B/32, chuẩn hóa và tính vector trung bình trước khi truy vấn FAISS IndexFlatIP.
- **Kết quả:** Khung hình mục tiêu xuất hiện tại Top 3 với độ tin cậy đạt 84.6% tại Video L27_V012 mốc 04:18 (Frame 6450). Thí sinh bấm nút '+' đưa thẳng vào danh sách nộp bài KIS.

### 3.2. Tình huống 2: Truy vấn Chuỗi Hành động theo Trình tự Thời gian (Đợt thi 28/08/2026)
- **Câu truy vấn:** *“Chiếc xe hơi màu đỏ dừng lại trước cổng -> người lái xe bước ra mở cốp xe”*
- **Đặc điểm & Độ khó:** Hành động trải dài qua nhiều keyframe liên tiếp trong một khoảng thời gian. Nếu chỉ tìm 'mở cốp xe', kết quả có thể ra xe màu khác hoặc ở bối cảnh khác.
- **Cách tiếp cận & Thuật toán:** Sử dụng cú pháp mũi tên `->`: Nhập `red car stops at gate -> driver opens trunk`. Backend gọi hàm `temporal_search`, lấy top 200 ứng viên của sự kiện 1 và top 200 sự kiện 2, sau đó ghép cặp trên cùng `video_id` thỏa mãn điều kiện thời gian: $0 < frame_2 - frame_1 \le 900$ (trong vòng 30 giây). Thí sinh dùng thêm Tab 'Frame Context' để kiểm tra toàn bộ diễn biến xung quanh.
- **Kết quả:** Hệ thống định vị chính xác cặp frame mục tiêu tại Video L18_V005: Frame 2150 (xe đỏ dừng) và Frame 2420 (mở cốp sau đó 10.8 giây). Điểm số tổng hợp đạt 89.2%.

### 3.3. Tình huống 3: Truy vấn Kết hợp Văn bản OCR / Biển số / Ký tự (Đợt thi 04/09/2026)
- **Câu truy vấn:** *“Xe cứu thương màu trắng di chuyển trên đường phố có dòng chữ CẤP CỨU hoặc số 115”*
- **Đặc điểm & Độ khó:** Dấu hiệu quyết định để tìm đúng video là chữ 'CẤP CỨU' hoặc số hiệu '115' in trên thân xe.
- **Cách tiếp cận & Thuật toán:** Thí sinh chọn chế độ 'OCR'. Hệ thống thực hiện tìm kiếm mờ (Fuzzy LIKE Matching) trên trường `ocr_text` của 69.160 keyframes trong SQLite. Kết quả được đối chiếu nhanh trong Inspection Modal với lớp phủ Bounding Box màu xanh hiển thị rõ vị trí chữ '115'.
- **Kết quả:** Tìm thấy chính xác video mục tiêu L21_V034 tại mốc 45.2 giây (Frame 1130). Bounding box hiển thị trực tiếp chữ '115' trên thân xe cứu thương với độ rõ nét cao.

### 3.4. Tình huống 4: Truy vấn Ngược từ Hình ảnh Mẫu (Visual Similarity / Image Search)
- **Câu truy vấn:** *“Tìm phân cảnh trong video chứa biểu trưng / vật thể có hình dáng tương tự ảnh cung cấp”*
- **Đặc điểm & Độ khó:** Đề thi cung cấp ảnh chụp hoặc cắt từ video, việc mô tả bằng lời rất khó nắm bắt được hoa văn và góc chụp.
- **Cách tiếp cận & Thuật toán:** Thí sinh nhấn Ctrl+V dán trực tiếp ảnh mẫu vào Tab 'Image Search'. Backend sinh vector 512 chiều qua `model.encode_image()` và thực hiện truy xuất FAISS IndexFlatIP trên 177.321 frames.
- **Kết quả:** Khung hình gốc trong video L29_V018 xuất hiện ngay tại vị trí Top 1 với độ tương đồng 96.8% chỉ sau 0.8 giây. Thí sinh sử dụng nút 'Similar' và 'Track' để xem tiếp các frame kế cận.

---

## 4. KẾT LUẬN VÀ CAM ĐOAN CỦA ĐỘI THI
Hệ thống Video Retrieval System V3 tham dự AI Challenge TP.HCM 2026 của đội HungryForSuccess(๑>•̀๑) là thành quả của quá trình nghiên cứu, tối ưu hóa nghiêm túc và thực chất. Hệ thống chứng minh rằng với việc làm chủ các kỹ thuật lập chỉ mục dữ liệu hiệu năng cao (FAISS, SQLite Local-First), thiết kế kiến trúc lưu trữ đám mây thông minh (Google Drive Proxy, Tampermonkey Jumper) và xây dựng giao diện tác chiến chuyên sâu, đội thi có thể đạt được hiệu suất truy xuất video vượt trội trong thời gian thực mà không cần phụ thuộc vào bất kỳ mô hình Generative AI hay dịch vụ bên ngoài nào.

Đội thi **HungryForSuccess(๑>•̀๑)** xin cam đoan rằng toàn bộ nội dung, cấu trúc hệ thống, các con số thống kê dữ liệu và kết quả thực nghiệm trình bày trong bản báo cáo này hoàn toàn phản ánh trung thực 100% hệ thống thực tế mà đội đã lập trình và vận hành.

*Thành phố Hồ Chí Minh, ngày 11 tháng 09 năm 2026*  
**ĐẠI DIỆN ĐỘI THI HUNGRYFORSUCCESS(๑>•̀๑)**  
*(Ký và ghi rõ họ tên)*  
**Phạm Huy (Trưởng nhóm)**
