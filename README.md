
# Nhận diện khuôn mặt bằng OpenCV (Sử dụng đặc trưng LBP)

Dự án này xây dựng một hệ thống **nhận diện khuôn mặt thời gian thực** bằng ngôn ngữ Java và thư viện OpenCV.  
Hệ thống sử dụng **Haar Cascade Classifier** để phát hiện khuôn mặt, **Local Binary Pattern (LBP)** để trích xuất đặc trưng, và so sánh bằng **khoảng cách Euclidean** để nhận diện người.

---

## 🧠 Thuật toán sử dụng

- **Phát hiện khuôn mặt (Face Detection):**  
  - Sử dụng mô hình học máy **Haar Cascade Classifier**.
  - Phát hiện khuôn mặt dựa trên đặc trưng cường độ pixel vùng lân cận.

- **Trích xuất đặc trưng (Feature Extraction):**  
  - Áp dụng thuật toán **Local Binary Pattern (LBP)** trên ảnh khuôn mặt.
  - LBP biến đổi đặc trưng kết cấu cục bộ thành một vector histogram, giúp tăng khả năng phân biệt khuôn mặt ngay cả khi thay đổi ánh sáng.

- **Nhận diện khuôn mặt (Face Recognition):**  
  - So sánh vector đặc trưng của khuôn mặt cần nhận diện với các vector lưu trữ trong dataset.
  - Tính **khoảng cách Euclidean** giữa các vector để tìm người có đặc trưng gần nhất.

---

## 📦 Cấu trúc thư mục dự án
```
├── dataset/                  # Thư mục chứa ảnh huấn luyện (mỗi người 1 thư mục riêng)
│    ├── person1/
│    │    ├── img1.jpg
│    │    ├── img2.jpg
│    ├── person2/
│    │    ├── img1.jpg
│    │    ├── img2.jpg
├── haarcascade_frontalface_default.xml    # File mô hình Haar Cascade
├── src/
│    ├── org/example/demo1/
│         ├── LBPFeatureExtractor.java     # Tệp trích xuất đặc trưng LBP
│         ├── nhandien.java                 # Chương trình chính
├── README.md
```

---

## 🛠 Hướng dẫn chi tiết cách cài đặt và chạy

### 1. Tải dự án về
```bash
git clone https://github.com/ten-cua-ban/ten-repo.git
cd ten-repo
```
(Thay `ten-cua-ban/ten-repo` bằng link GitHub của bạn.)

---

### 2. Cài đặt OpenCV cho Java
- Tải [OpenCV 4.x](https://opencv.org/releases/).
- Giải nén vào một thư mục, ví dụ: `C:/opencv`.

- Thêm file `.jar` của OpenCV vào project:
  ```
  C:/opencv/build/java/opencv-430.jar
  ```

- Cấu hình đường dẫn thư viện native:
  - Trong VM Options, thêm:
    ```
    -Djava.library.path=C:/opencv/build/java/x64
    ```
  *(Điều chỉnh đường dẫn theo phiên bản và máy của bạn.)*

---

### 3. Chuẩn bị dữ liệu dataset
- Tạo thư mục `dataset/` ngay trong project.
- Tạo **một thư mục cho mỗi người** bạn muốn nhận diện.
- Bên trong mỗi thư mục, thêm **nhiều ảnh khuôn mặt** (nên dùng ảnh rõ ràng).

Ví dụ:
```
dataset/
├── Alice/
│    ├── img1.jpg
│    ├── img2.jpg
├── Bob/
│    ├── img1.jpg
│    ├── img2.jpg
```
**Lưu ý:**  
- Ảnh nên ở **chế độ grayscale** (ảnh đen trắng).  
- Khuôn mặt nên **rõ nét, đủ sáng**, và chiếm phần lớn bức ảnh.

---

### 4. Tải file Haar Cascade
- Tải [`haarcascade_frontalface_default.xml`](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).
- Đặt file này cùng cấp với thư mục `dataset/`.

---

### 5. Biên dịch và chạy chương trình
- Mở project trong IDE (IntelliJ IDEA, Eclipse...).
- Đảm bảo đã add OpenCV `.jar` vào project.
- Chạy chương trình `org.example.demo1.nhandien`.

---

### 6. Sử dụng chương trình
- Webcam sẽ tự động bật lên.
- Khi phát hiện khuôn mặt:
  - Vẽ khung xanh quanh khuôn mặt.
  - Hiển thị **tên** và **mức độ tin cậy** (%) trên khuôn mặt.
- Nhấn **`q`** để thoát chương trình.

---

## 🧠 Một số lỗi thường gặp
| Vấn đề | Cách khắc phục |
|:---|:---|
| Không mở được webcam | Kiểm tra thiết bị hoặc quyền truy cập camera. Thử đổi `new VideoCapture(0)` thành `1`. |
| Lỗi `UnsatisfiedLinkError` | Kiểm tra lại tham số `-Djava.library.path`. |
| Không phát hiện được khuôn mặt | Xem lại đường dẫn XML đúng chưa. Đảm bảo ảnh đủ sáng và rõ. |
| Sai người nhận diện | Bổ sung nhiều ảnh hơn cho mỗi người. |

---

## 📚 Tài liệu tham khảo
- [Tài liệu chính thức OpenCV](https://docs.opencv.org/4.x/)
- [Giải thích Haar Cascade](https://docs.opencv.org/4.x/d7/d8b/tutorial_py_face_detection.html)
- [Giới thiệu Local Binary Patterns (LBP)](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_local_binary_pattern.html)

---

## 📸 Demo minh họa chương trình
> *()*

---

