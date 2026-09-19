# Bản Kế Hoạch Nâng Cấp Toàn Diện Hệ Thống Video Retrieval

Bản kế hoạch này tổng hợp toàn bộ các phương án tối ưu, nâng cấp giao diện và tích hợp AI thông minh để biến dự án thành một hệ thống tra cứu chuyên nghiệp nhất. Kế hoạch được chia thành 4 giai đoạn thực hiện.

---

## 🛠️ Giai đoạn 1: Tối ưu Tốc độ Hiển thị (Cloudflare R2)
*Trạng thái: Đang tiến hành (In-progress)*
**Mục tiêu:** Loại bỏ sự chậm chạp của Google Drive, giúp ảnh tải lên màn hình ngay lập tức (dưới 1 giây).
- **Hạng mục 1.1:** Dùng script thu nhỏ 177K ảnh gốc thành thumbnail và nén thành định dạng `WEBP` (~15KB/ảnh).
- **Hạng mục 1.2:** Tải toàn bộ 2.6GB ảnh đã nén lên hệ thống Cloudflare R2 (Bản Free 10GB).
- **Hạng mục 1.3:** Xóa mã Proxy Google Drive trong Backend, cập nhật Frontend để load thẳng ảnh từ link CDN của Cloudflare.

---

## 🎤 Giai đoạn 2: Nâng cấp Trải nghiệm Người dùng (Voice Search)
*Trạng thái: Đã hoàn tất (Completed) ✅*
**Mục tiêu:** Rảnh tay gõ phím, tăng tốc độ nhập liệu bằng thao tác giọng nói.
- **Hạng mục 2.1:** Thêm nút Micro vào thanh tìm kiếm trên giao diện Web (`index.html`).
- **Hạng mục 2.2:** Tích hợp `Web Speech API` nhận diện giọng nói (Tiếng Việt/Anh) cực nhạy.
- **Hạng mục 2.3:** Lập trình tự động kích hoạt tiến trình Search ngay khi người dùng ngừng nói.

---

## 🤖 Giai đoạn 3: Nhúng Cửa sổ AI Trợ lý (Bong bóng Chat Kéo Thả)
*Trạng thái: Chờ thực hiện*
**Mục tiêu:** Người dùng vừa có thể tự tìm kiếm bằng tay, vừa có thể nhờ AI tìm kiếm hộ mà không bị "đập" giao diện của nhau.
- **Hạng mục 3.1:** Code một Bong bóng Chat nổi (Floating Widget) ở góc màn hình, có thể dùng chuột kéo thả tự do.
- **Hạng mục 3.2:** Tích hợp Antigravity SDK vào Backend (`FastAPI`) để nuôi một bộ não AI chạy ngầm.
- **Hạng mục 3.3:** Cơ chế cách ly: AI trả kết quả (Text + Ảnh thu nhỏ) ngay trong Bong bóng Chat. Chỉ khi người dùng bấm "Mở rộng", ảnh mới được chèn ra màn hình lưới (Gallery) chính.

---

## 🧠 Giai đoạn 4: Đào tạo Kỹ năng "Chuyên gia Truy Xuất Video" cho AI
*Trạng thái: Chờ thực hiện*
**Mục tiêu:** Tạo bộ não (System Prompt/Skill) bài bản, giúp AI tự biết cách tìm kiếm thông minh, đỡ tốn thời gian Training của người dùng.
- **Hạng mục 4.1 - Phân rã sự kiện (Temporal Reasoning):** Dạy AI cách tìm hành động A, sau đó tự tua tới/lui (dùng tool `get_frame_context`) để tìm hành động B.
- **Hạng mục 4.2 - Tìm kiếm chéo (Cross-Modal):** Dạy AI tự động kết hợp đối chiếu hình ảnh (`Semantic`) với lời thoại (`ASR`) để chốt kết quả chuẩn xác nhất.
- **Hạng mục 4.3 - Tư duy chuyển dịch:** Dạy AI biết tự động đổi các từ khóa trừu tượng (như "đau buồn") thành hình ảnh thực tế ("người khóc") nếu lần tìm kiếm đầu tiên bị thất bại.
- **Hạng mục 4.4 - Báo cáo cô đọng:** Quy định AI không được xả rác data. Chỉ lọc ra 1 đến 3 khung hình tuyệt đối chính xác nhất và báo cáo lý do vô cùng ngắn gọn.
