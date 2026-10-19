# 🏢 Smart Attendance System (AI-Powered Face Recognition)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/Streamlit-1.55.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/OpenCV-4.13.0-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/dlib-128D_Embeddings-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="dlib" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/Docker_Compose-Multi--Container-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Compose" />
  <img src="https://img.shields.io/badge/SQLite-SQLAlchemy_2.0-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite" />
  <img src="https://img.shields.io/badge/UI/UX-Clean_Enterprise_SaaS-1E3A8A?style=for-the-badge" alt="Enterprise Design System" />
</p>

<p align="center">
  <strong>Hệ thống điểm danh và quản trị nhân sự thông minh ứng dụng Thị giác máy tính (Computer Vision) & Deep Metric Learning thời gian thực.</strong>
</p>

---

## 📑 Mục Lục (Table of Contents)
1. [Giới Thiệu Dự Án (Project Overview)](#-giới-thiệu-dự-án-project-overview)
2. [Các Tính Năng Nổi Bật (Key Highlights)](#-các-tính-năng-nổi-bật-key-highlights)
3. [Cấu Trúc Thư Mục Dự Án (Project Structure)](#-cấu-trúc-thư-mục-dự-án-project-structure)
4. [Công Nghệ Sử Dụng (Tech Stack)](#️-công-nghệ-sử-dụng-tech-stack)
5. [Sơ Đồ Kiến Trúc Hệ Thống (System Architecture)](#️-sơ-đồ-kiến-trúc-hệ-thống-system-architecture)
6. [Cài Đặt & Triển Khai (Installation & Deployment)](#-cài-đặt--khởi-chạy-installation--quick-start)
   - [Phương Thức 1: Triển Khai Với Docker & Docker Compose](#-phương-thức-1-triển-khai-nhanh-với-docker--docker-compose-khuyến-nghị)
   - [Phương Thức 2: Cài Đặt Thủ Công Bằng Virtualenv](#-phương-thức-2-cài-đặt-thủ-công-bằng-virtualenv-local-development)
7. [Hướng Dẫn Sử Dụng (User Manual)](#-hướng-dẫn-sử-dụng-chi-tiết-user-manual--workflows)
8. [Kiểm Thử Tự Động (Automated Testing)](#-kiểm-thử-tự-động-automated-testing)
9. [Tiêu Chuẩn Thiết Kế & An Toàn Sinh Trắc Học (Design & Security)](#-tiêu-chuẩn-bảo-mật--quyền-riêng-tư-sinh-trắc-học-security--biometric-privacy)

---

## 🌟 Giới Thiệu Dự Án (Project Overview)

**Smart Attendance System** là giải pháp phần mềm điểm danh doanh nghiệp hiện đại được xây dựng trên nền tảng **Python, OpenCV, dlib và Streamlit**. Hệ thống giải quyết trọn vẹn bài toán ghi nhận công tự động thông qua camera AI:

- **Loại bỏ thẻ từ và vân tay**: Sử dụng khuôn mặt làm định danh sinh trắc học không tiếp xúc, phòng ngừa gian lận điểm danh hộ (buddy punching).
- **Trải nghiệm mượt mà không độ trễ**: Tích hợp kiến trúc luồng đọc camera chuyên dụng đa luồng (Multi-Threaded DirectShow Capture) triệt tiêu 100% hiện tượng trễ hình (Zero-Delay Streaming).
- **Logic kiểm soát 2 chiều nghiêm ngặt**: Phân định rành mạch cổng **VÀO (IN)** và cổng **RA (OUT)**, chống ghi nhận trùng lặp với cơ chế Debounce Cooldown và xác thực 3-frame ổn định.
- **Chuẩn giao diện doanh nghiệp cao cấp**: Thiết kế Clean Enterprise SaaS UI với Typography phân cấp sắc nét, hệ màu Design Tokens ngữ nghĩa (`#10B981` cho IN, `#F59E0B` cho OUT, `#1E3A8A` Brand Corporate).

---

## 🚀 Các Tính Năng Nổi Bật (Key Highlights)

### 1. 📊 Bảng Điều Khiển Trung Tâm (Attendance Dashboard)
- Hiển thị tức thì các chỉ số KPI: **Tổng nhân sự**, **Số lượt Check In hôm nay**, **Số lượt Check Out hôm nay**.
- Trực quan hóa dữ liệu biểu đồ cột tương tác Plotly thể hiện xu hướng điểm danh trong **7 ngày gần nhất**.
- Khung **Latest Activity** hiển thị card thông tin nhân viên vừa được hệ thống AI nhận diện theo thời gian thực.
- Bảng nhật ký điểm danh trong ngày tích hợp bộ lọc trực tiếp theo mã nhân viên và trạng thái (IN/OUT).

### 2. 👥 Quản Lý Hồ Sơ Nhân Viên (Employee Management)
- Quản lý đầy đủ thông tin nhân sự: Mã nhân viên (Employee Code), Họ tên, Phòng ban, Chức vụ.
- Hỗ trợ thêm mới, chỉnh sửa thông tin, xác thực trường dữ liệu và xóa nhân sự an toàn.

### 3. 🧑‍💻 Thu Thập & Đăng Ký Dữ Liệu Khuôn Mặt (Register Face)
- Thu thập mẫu ảnh khuôn mặt trực tiếp từ Native Webcam theo thời gian thực hoặc tải lên ảnh có sẵn.
- Tự động phát hiện khuôn mặt, chuẩn hóa tỷ lệ khung cắt (Crop & Resize) và phân loại theo từng thư mục định danh nhân viên.

### 4. 📷 Trạm Camera Điểm Danh Tự Động (Check-in Camera - Native Live)
- Hỗ trợ đồng thời 2 cấu hình camera:
  - **Camera 1 (Cổng IN)**: Chuyên trách ghi nhận nhân viên vào làm.
  - **Camera 2 (Cổng OUT)**: Chuyên trách ghi nhận nhân viên ra về.
- Cơ chế nhận diện an toàn:
  - **Target Locked**: Theo dõi và yêu cầu tối thiểu **3 frames ổn định liên tiếp** trước khi chốt giao dịch điểm danh.
  - **Toast Notifications Real-time**: Bật popup thông báo tức thì ngay trên màn hình streaming mà không làm giật/đứng khung hình.
  - **Ràng buộc nghiệp vụ chặt chẽ**: Từ chối Check Out nếu chưa Check In; cảnh báo nếu nhân viên đã điểm danh trong ca.

### 5. 🕒 Lịch Sử & Xuất Báo Cáo (History & Export)
- Tra cứu toàn bộ lịch sử điểm danh theo thời gian thực với bộ lọc đa tiêu chí (Mã nhân viên, Loại điểm danh IN/OUT).
- Xuất dữ liệu báo cáo chuyên nghiệp sang định dạng **Excel (.xlsx)** hoặc **CSV** phục vụ công tác tính công và tính lương.

### 6. 🔍 Phân Hệ Kiểm Thử AI (AI Test Bench)
- **Face Detection (Test)**: Kiểm tra trực quan thuật toán phát hiện vị trí khuôn mặt và trích xuất tọa độ Bounding Box.
- **Face Recognition (Test)**: Kiểm thử độ chính xác của vector 128D embedding, khoảng cách Euclidean/Cosine và ngưỡng phân loại.

---

## 📂 Cấu Trúc Thư Mục Dự Án (Project Structure)

Toàn bộ dự án được tổ chức theo kiến trúc phân tầng rõ ràng (Clean Architecture), tách biệt giữa tầng xử lý thị giác máy tính (`core/`), tầng dữ liệu (`db/`), tầng giao diện người dùng (`ui/`, `pages/`) và kiểm thử (`tests/`):

```text
Do_an_AI/
|-- Dockerfile                 # Đóng gói môi trường container (Python 3.11, C++, dlib, OpenCV)
|-- docker-compose.yml         # Cấu hình dịch vụ, port mapping 8501, volume mount dữ liệu
|-- .dockerignore              # Loại trừ tệp tạm, venv và git cache khi build Docker image
|-- app.py                     # Entry point chính của ứng dụng Streamlit (Navigation & Setup)
|-- requirements.txt           # Danh mục thư viện phụ thuộc (OpenCV, Streamlit, dlib, SQLAlchemy...)
|-- README.md                  # Tài liệu hướng dẫn dự án chuẩn Enterprise
|-- download_dataset.py        # Kịch bản tải hoặc khởi tạo bộ dữ liệu khuôn mặt mẫu
|-- .gitignore                 # Cấu hình loại trừ cache, virtual environment và dữ liệu nhạy cảm
|
|-- core/                      # TẦNG XỬ LÝ THỊ GIÁC MÁY TÍNH & AI CỐT LÕI
|   |-- camera_stream.py       # Đa luồng đọc camera DirectShow 0ms delay, quản lý vòng đời camera
|   |-- face_detect.py         # Phát hiện khuôn mặt (Haar Cascade & Face Bounding Box)
|   |-- face_recognizer.py     # Trích xuất 128D vector embedding (dlib) & đối sánh khoảng cách
|   |-- face_dataset.py        # Nạp dữ liệu ảnh, tiền xử lý và lập bản đồ nhãn huấn luyện
|   `-- save_face.py           # Tiền xử lý, chuẩn hóa kích thước và lưu trữ ảnh mẫu khuôn mặt
|
|-- db/                        # TẦNG CƠ SỞ DỮ LIỆU & REPOSITORY
|   |-- attendance.db          # Cơ sở dữ liệu cục bộ SQLite
|   |-- models.py              # Định nghĩa schema ORM SQLAlchemy (Employees, AttendanceLogs)
|   |-- session.py             # Quản lý phiên kết nối DB Session
|   |-- database.py            # CRUD Repository cho nghiệp vụ Nhân viên (Employees)
|   `-- attendance_repo.py     # Repository xử lý giao dịch điểm danh & Business Rules (IN/OUT)
|
|-- pages/                     # TẦNG GIAO DIỆN NGƯỜI DÙNG STREAMLIT (MULTIPAGE APP)
|   |-- 1_Dashboard.py         # Bảng điều khiển phân tích số liệu, biểu đồ 7 ngày & Latest Activity
|   |-- 2_Employees.py         # Quản lý hồ sơ nhân viên, chức vụ, phòng ban
|   |-- 3_Register_Face.py     # Đăng ký và chụp ảnh khuôn mặt nhân viên thời gian thực
|   |-- 4_Attendance.py        # Trạm camera quét điểm danh tự động 2 cổng IN/OUT với Toast real-time
|   |-- 5_History.py           # Tra cứu lịch sử điểm danh và xuất báo cáo Excel/CSV
|   |-- 6_Face_Detection.py    # Trang kiểm thử thuật toán phát hiện khuôn mặt
|   `-- 7_Face_Recognition.py  # Trang kiểm thử đối sánh nhận diện & đo khoảng cách vector AI
|
|-- ui/                        # TẦNG GIAO DIỆN & DESIGN SYSTEM
|   `-- styles.py              # Custom CSS toàn cục (Typography, Container, Tokens, Sidebar toggle)
|
|-- models/                    # LƯU TRỮ TRỌNG SỐ VÀ ENCODINGS AI
|   |-- face_encodings.pkl     # File nhị phân lưu trữ tập 128D embeddings của toàn bộ nhân viên
|   |-- label_map.json         # Ánh xạ chỉ số nhãn sang mã nhân viên
|   `-- model_metadata.json    # Siêu dữ liệu mô hình (ngưỡng threshold, phiên bản huấn luyện)
|
|-- data/                      # THƯ MỤC LƯU TRỮ DỮ LIỆU CỤC BỘ
|   `-- faces/                 # Tập thư mục ảnh mẫu khuôn mặt phân theo mã nhân viên
|
`-- tests/                     # BỘ KIỂM THỬ TỰ ĐỘNG (AUTOMATED TEST SUITE)
    |-- test_db.py             # Kiểm thử kết nối và CRUD cơ sở dữ liệu
    |-- test_phase1_streaming.py # Kiểm thử bộ đệm đa luồng DirectShow camera
    |-- test_phase1_attendance_toast.py # Kiểm thử cơ chế bắn Toast real-time & debounce
    |-- test_phase2_attendance.py # Kiểm thử ràng buộc nghiệp vụ điểm danh 2 chiều
    `-- test_phase3_face_recognition.py # Kiểm thử độ chính xác trích xuất vector và đối sánh AI
```

---

## 🛠️ Công Nghệ Sử Dụng (Tech Stack)

| Phân hệ | Công nghệ | Mục đích |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python 3.11+ | Nền tảng phát triển ứng dụng chính |
| **Container hóa** | Docker & Docker Compose | Đóng gói môi trường, quản lý service và dữ liệu độc lập nền tảng |
| **Giao diện Web** | Streamlit 1.55+ | Xây dựng giao diện Web Multipage tương tác cao |
| **Thị giác máy tính** | OpenCV 4.13 | Bắt luồng camera DirectShow đa luồng, xử lý khung hình |
| **Deep Learning** | `dlib` & `face_recognition` | Trích xuất 128 chiều đặc trưng khuôn mặt (Deep Metric ResNet) |
| **Biểu đồ & Thống kê**| Plotly Express & Pandas | Trực quan hóa dữ liệu xu hướng điểm danh và xử lý bảng biểu |
| **Cơ sở dữ liệu** | SQLite & SQLAlchemy 2.0 | Lưu trữ hồ sơ nhân viên và lịch sử điểm danh giao dịch ACID |
| **Xuất báo cáo** | OpenPyXL | Kết xuất báo cáo định dạng bảng tính Excel (.xlsx) |

---

## 🏛️ Sơ Đồ Kiến Trúc Hệ Thống (System Architecture)

### 1. Luồng Xử Lý Thị Giác Máy Tính (AI Vision Pipeline)

Hệ thống kết hợp giữa **OpenCV DirectShow đa luồng** để giải phóng bộ đệm camera và mạng học sâu **dlib ResNet-34 (128D Embedding)** để đối sánh khuôn mặt đạt độ chính xác cao:

```mermaid
flowchart TD
    subgraph VideoCapture ["1. Luồng Thu Nhận Hình Ảnh (Zero-Delay)"]
        Cam["Webcam / IP Camera"] --> DirectShow["DirectShow Bufferless Thread (ThreadedCameraReader)"]
        DirectShow --> FrameRGB["Khung hình chuẩn RGB (640x480 @ 30fps)"]
    end

    subgraph FaceProcessing ["2. Tiền Xử Lý & Trích Xuất Đặc Trưng AI"]
        FrameRGB --> HaarDetect["Phát hiện khuôn mặt (Haar Cascade / dlib)"]
        HaarDetect --> FaceAlign["Căn chỉnh & Crop vùng mặt (Tỷ lệ đệm 18%)"]
        FaceAlign --> DlibResNet["Mạng ResNet-34 trích xuất vector 128 chiều"]
        DlibResNet --> Vector128D["128-Dimensional Face Embedding Vector"]
    end

    subgraph MatchingEngine ["3. Động Cơ Đối Sánh & Phân Loại"]
        Vector128D --> DistanceCalc["Tính khoảng cách Euclidean / Cosine"]
        PreTrained[("Tập Embeddings Nhân viên (face_encodings.pkl)")] --> DistanceCalc
        DistanceCalc --> ThresholdCheck{"Khoảng cách <= Ngưỡng an toàn (0.48)?"}
        ThresholdCheck -- "Không" --> Unknown["Gán nhãn Unknown Employee"]
        ThresholdCheck -- "Có" --> MatchFound["Xác định Mã nhân viên (Target Locked)"]
    end

    subgraph BusinessLogic ["4. Nghiệp Vụ Điểm Danh & Phản Hồi UI"]
        MatchFound --> Stability{"Tích lũy đủ 3 frames ổn định?"}
        Stability -- "Chờ thêm frame" --> StabilizingUI["Hiển thị thanh tiến trình Stabilizing..."]
        Stability -- "Đạt chuẩn" --> AttendanceRules{"Kiểm tra quy tắc (Cổng IN / Cổng OUT)"}
        AttendanceRules --> Transaction[("Ghi nhận giao dịch SQLite ACID (attendance.db)")]
        Transaction --> ToastAlert["Kích hoạt Streamlit Toast Notification (Debounced)"]
        ToastAlert --> RealtimeUI["Cập nhật Dashboard & Latest Activity Real-time"]
    end
```

---

### 2. Quy Tắc Nghiệp Vụ Điểm Danh Hai Chiều (Attendance Business Logic)

Để đảm bảo dữ liệu chấm công minh bạch và chống gian lận, hệ thống áp dụng bộ quy tắc kiểm soát logic giao dịch nghiêm ngặt:

```mermaid
flowchart LR
    A["Camera nhận diện nhân viên (Code X)"] --> B{"Kiểm tra Cổng Quét"}
    
    B -- "Cổng 1: Check In (IN)" --> C{"Nhân viên đã Check In hôm nay chưa?"}
    C -- "Đã Check In rồi" --> C1["⚠️ Từ chối: Đã check-in rồi! Bắn Toast cảnh báo"]
    C -- "Chưa Check In" --> C2["✅ Ghi nhận Check In thành công (Màu xanh #10B981)"]

    B -- "Cổng 2: Check Out (OUT)" --> D{"Nhân viên đã Check In hôm nay chưa?"}
    D -- "Chưa từng Check In" --> D1["⚠️ Từ chối: Chưa check-in hôm nay, không thể check-out!"]
    D -- "Đã Check In" --> E{"Đã Check Out hôm nay chưa?"}
    E -- "Đã Check Out rồi" --> E1["⚠️ Từ chối: Đã check-out ca làm việc rồi!"]
    E -- "Chưa Check Out" --> E2["✅ Ghi nhận Check Out thành công (Màu cam #F59E0B)"]
```

#### Các Cơ Chế Chống Lỗi & Tối Ưu Trải Nghiệm (Fail-Safe & UX Protections):
1. **Target Locked (3 Frames Stability)**:
   - Người đứng trước camera cần giữ yên vị trí để hệ thống nhận diện ổn định trong **3 frames liên tiếp** cùng một mã nhân viên.
   - Tránh hiện tượng camera ghi nhận nhầm khi nhân viên chỉ vô tình đi lướt ngang qua vùng quét.
2. **Debounce Cooldown (Chống Spam Điểm Danh)**:
   - Sau khi ghi nhận thành công, camera kích hoạt thời gian chờ **30 giây** đối với nhân viên đó (`AUTO_ACTION_COOLDOWN_SECONDS = 30`).
   - Ngăn chặn triệt để việc một người đứng trước ống kính làm sinh ra hàng chục bản ghi liên tiếp.
3. **Quản Lý Vòng Đời Thiết Bị (`release_inactive_cameras`)**:
   - Khi chuyển trang trong Streamlit, hệ thống tự động ngắt kết nối và giải phóng camera phần cứng, bảo đảm không bị chiếm dụng camera (Camera Resource Conflict).

---

### 3. Bảng Tham Số Kỹ Thuật (System Hyperparameters)

Dưới đây là các tham số cấu hình cốt lõi được định nghĩa trong [core/camera_stream.py](file:///d:/Do_an_AI/core/camera_stream.py), [core/face_recognizer.py](file:///d:/Do_an_AI/core/face_recognizer.py) và [pages/4_Attendance.py](file:///d:/Do_an_AI/pages/4_Attendance.py):

| Tham số | Giá trị | Tệp khai báo | Mục đích giải thích |
| :--- | :---: | :--- | :--- |
| `DEFAULT_MATCH_THRESHOLD` | `0.48` | `core/face_recognizer.py` | Ngưỡng khoảng cách Euclidean tối đa để chấp nhận khuôn mặt là trùng khớp (Khoảng cách giữa 2 người khác nhau luôn $\ge 0.4917$). |
| `STRICT_MATCH_THRESHOLD` | `0.44` | `core/face_recognizer.py` | Ngưỡng nhận diện nghiêm ngặt trong điều kiện ánh sáng phức tạp. |
| `FACE_SIZE` | `(200, 200)` | `core/save_face.py` | Kích thước ảnh khuôn mặt chuẩn hóa sau khi crop để lưu trữ và huấn luyện. |
| `FACE_PADDING_RATIO` | `0.18` (18%) | `core/save_face.py` | Tỷ lệ mở rộng khung cắt viền khuôn mặt để giữ lại trọn vẹn cằm và trán. |
| `AUTO_MATCH_REQUIRED_FRAMES` | `3` frames | `pages/4_Attendance.py` | Số khung hình liên tiếp bắt buộc nhận diện cùng một người trước khi chốt điểm danh. |
| `AUTO_ACTION_COOLDOWN_SECONDS` | `30` giây | `pages/4_Attendance.py` | Thời gian đóng băng điểm danh cùng loại (IN/OUT) đối với một nhân viên để chống trùng lặp. |
| `TOAST_COOLDOWN_SECONDS` | `5.0` giây | `pages/4_Attendance.py` | Thời gian giãn cách giữa các thông báo Toast trên màn hình camera. |
| `STREAM_FRAME_INTERVAL` | `0.033` giây | `core/camera_stream.py` | Chu kỳ cập nhật khung hình hiển thị tương đương **~30 FPS** mượt mà. |
| `OMP_NUM_THREADS` | `2` | `core/camera_stream.py` | Giới hạn số luồng tính toán song song của Dlib/OpenMP nhằm giải phóng CPU cho UI. |

---

## 💻 Cài Đặt & Khởi Chạy (Installation & Quick Start)

### 1. Yêu Cầu Hệ Thống (Prerequisites)
- **Hệ điều hành**: Windows 10/11, Ubuntu/Debian Linux hoặc macOS.
- **Tùy chọn A (Triển khai bằng Docker - Khuyến nghị)**: Đã cài đặt [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows/macOS) hoặc Docker Engine & Docker Compose (Linux).
- **Tùy chọn B (Cài đặt thủ công)**: Python `3.10` hoặc `3.11` (khuyến nghị **Python 3.11.x**). Trên Windows cần cài [Visual Studio Build Tools C++](https://visualstudio.microsoft.com/) để biên dịch `dlib`.
- **Phần cứng**: Webcam USB hoặc Camera tích hợp Laptop.

---

### 2. Phương Thức 1: Triển Khai Nhanh Với Docker & Docker Compose (Khuyến nghị)

Triển khai bằng Docker giúp đóng gói toàn bộ thư viện C++, Dlib và OpenCV mà **không cần cài đặt môi trường C++ phức tạp** trên máy chủ:

#### Bước 1: Sao chép mã nguồn (Clone Repository)
```bash
git clone https://github.com/your-username/Do_an_AI.git
cd Do_an_AI
```

#### Bước 2: Khởi chạy dịch vụ với Docker Compose
```bash
# Tự động build image và khởi chạy container ngầm (Detached mode)
docker compose up --build -d
```

#### Bước 3: Kiểm tra trạng thái và theo dõi Log
```bash
# Xem log hoạt động của dịch vụ Streamlit thời gian thực
docker compose logs -f attendance-system

# Kiểm tra trạng thái container và healthcheck
docker compose ps
```

Sau khi container khởi động thành công, truy cập giao diện điểm danh tại: **`http://localhost:8501`**.

#### Bước 4: Dừng dịch vụ
```bash
# Dừng và dọn dẹp container (Dữ liệu SQLite và ảnh khuôn mặt vẫn được lưu an toàn)
docker compose down
```

---

#### 💡 Tùy Chọn: Chạy Bằng Lệnh `docker run` Độc Lập
Nếu không sử dụng Docker Compose, bạn có thể build và chạy trực tiếp bằng Docker CLI:
```bash
# 1. Build Docker image
docker build -t smart-face-attendance:latest .

# 2. Khởi chạy container với Persistent Volume mounts
docker run -d \
  --name smart_face_attendance \
  -p 8501:8501 \
  -v "$(pwd)/data/faces:/app/data/faces" \
  -v "$(pwd)/models:/app/models" \
  -v "$(pwd)/attendance.db:/app/attendance.db" \
  --restart unless-stopped \
  smart-face-attendance:latest
```

---

#### ⚙️ Các Lưu Ý Kỹ Thuật Quan Trọng Về Docker (Docker Best Practices):
1. **Bảo Toàn Dữ Liệu Bền Vững (Persistent Volumes)**:
   - File [docker-compose.yml](file:///d:/Do_an_AI/docker-compose.yml) đã cấu hình mount các thư mục quan trọng ra máy chủ (Host):
     - `./data/faces:/app/data/faces`: Lưu trữ tập ảnh khuôn mặt nhân viên đã thu thập.
     - `./models:/app/models`: Lưu trữ trọng số nhị phân `face_encodings.pkl` và bản đồ nhãn `label_map.json`.
     - `./attendance.db:/app/attendance.db`: Cơ sở dữ liệu SQLite lưu hồ sơ nhân sự và lịch sử điểm danh.
   - Khi bạn dừng (`docker compose down`) hoặc xóa container, **toàn bộ dữ liệu chấm công và ảnh nhận diện đều được bảo toàn 100%**.
2. **Cấu Hình Thiết Bị Webcam Trên Linux (Hardware Passthrough)**:
   - Khi triển khai trên máy chủ Linux (Ubuntu/Debian) có cắm webcam USB trực tiếp, mở comment dòng `devices` trong file [docker-compose.yml](file:///d:/Do_an_AI/docker-compose.yml):
     ```yaml
     devices:
       - "/dev/video0:/dev/video0"
     ```
   - Lệnh này cấp quyền cho container truy cập trực tiếp vào thiết bị camera phần cứng của hệ điều hành.
3. **Cơ Chế Giám Sát Sức Khỏe (Healthcheck)**:
   - `Dockerfile` tích hợp sẵn cơ chế kiểm tra định kỳ 30 giây qua endpoint:
     `curl -f http://localhost:8501/_stcore/health`
   - Đảm bảo phát hiện và khởi động lại kịp thời nếu máy chủ Streamlit gặp sự cố.

---

### 3. Phương Thức 2: Cài Đặt Thủ Công Bằng Virtualenv (Local Development)

Dành cho nhà phát triển muốn trực tiếp sửa code Python trên máy tính cục bộ:

#### Bước 1: Khởi tạo và kích hoạt môi trường ảo (Virtual Environment)
- **Trên Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Trên Linux / macOS**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### Bước 2: Cài đặt các gói thư viện phụ thuộc
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> [!NOTE]
> Danh mục thư viện chính trong `requirements.txt` bao gồm:
> `streamlit==1.55.0`, `opencv-python==4.13.0.92`, `numpy==2.4.3`, `pandas==2.3.3`, `SQLAlchemy>=2.0.0`, `plotly`, `dlib`, `face-recognition`, `openpyxl`.

#### Bước 3: Khởi chạy máy chủ Streamlit
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`.

---

## 📖 Hướng Dẫn Sử Dụng Chi Tiết (User Manual & Workflows)

### Luồng 1: Thêm Nhân Viên Mới & Đăng Ký Dữ Liệu Khuôn Mặt
1. Vào menu **Quản lý (Management)** $\to$ Chọn **Employees**: Nhập thông tin (Mã nhân viên `NV001`, Họ tên, Phòng ban, Chức vụ) và bấm **Create Employee**.
2. Chuyển sang menu **Nhận diện AI (AI Recognition)** $\to$ Chọn **Register Face**:
   - Chọn mã nhân viên tương ứng từ danh sách thả xuống.
   - Bật camera trực tiếp, căn chỉnh góc mặt thẳng, nghiêng nhẹ và bấm **Capture Face** (khuyến nghị thu thập từ 15 - 30 ảnh mẫu chuẩn).
   - Hệ thống tự động cắt khung mặt chuẩn 200x200 và lưu trữ an toàn trong thư mục `data/faces/<employee_code>/`.

### Luồng 2: Điểm Danh Tự Động Tại Trạm Camera
1. Chọn **Check-in Camera** trong danh mục **Nhận diện AI**:
   - Lựa chọn cổng chức năng tương ứng:
     - **Camera 1 (IN) 📥**: Dành cho cổng vào buổi sáng.
     - **Camera 2 (OUT) 📤**: Dành cho cổng ra ca về.
   - Bấm **Start Camera** để kích hoạt luồng camera DirectShow thời gian thực.
2. Nhân viên đứng trước camera:
   - Hệ thống nhận diện và khóa mục tiêu (**Target Locked**).
   - Tiến trình ổn định đạt **3/3 frames** $\to$ Tự động chốt điểm danh và bắn **Toast Notification** thông báo tức thì trên màn hình.
   - Thẻ log hiển thị trực quan thông tin: Mã nhân viên, Cổng, Thời gian và Độ tin cậy AI.

### Luồng 3: Giám Sát Dashboard & Xuất Báo Cáo Chấm Công
1. Truy cập **Dashboard**: Theo dõi tổng quan số nhân viên đi làm hôm nay, tỷ lệ có mặt và biểu đồ cột xu hướng 7 ngày.
2. Truy cập **History**:
   - Chọn bộ lọc theo mã nhân viên hoặc trạng thái (Tất cả / IN / OUT).
   - Bấm nút **Export to Excel (.xlsx)** để tải về bảng tổng hợp chấm công chi tiết.

---

## 🧪 Kiểm Thử Tự Động (Automated Testing)

Dự án cung cấp bộ kiểm thử toàn diện đặt trong thư mục `tests/`:

```powershell
# Chạy kiểm thử tương tác cơ sở dữ liệu (SQLite CRUD & Attendance Transactions)
$env:PYTHONPATH="."; python tests/test_db.py

# Chạy kiểm thử cơ chế bộ đệm đa luồng DirectShow camera
$env:PYTHONPATH="."; python tests/test_phase1_streaming.py

# Chạy kiểm thử Toast Notification thời gian thực & Debounce Cooldown
$env:PYTHONPATH="."; python tests/test_phase1_attendance_toast.py

# Chạy kiểm thử logic nghiệp vụ ràng buộc điểm danh 2 chiều (IN/OUT)
$env:PYTHONPATH="."; python tests/test_phase2_attendance.py

# Chạy kiểm thử trích xuất 128D Embedding và ngưỡng nhận diện Dlib
$env:PYTHONPATH="."; python tests/test_phase3_face_recognition.py
```

---

## 🔒 Tiêu Chuẩn Bảo Mật & Quyền Riêng Tư Sinh Trắc Học (Security & Biometric Privacy)

- **Biometric Template Protection**: Hệ thống chuyển đổi ảnh khuôn mặt thành vector 128 chiều (Face Embeddings) để so khớp khoảng cách toán học, hạn chế tối đa việc truyền tải hay lưu trữ ảnh khuôn mặt thô.
- **Git Hygiene**:
  - Toàn bộ cơ sở dữ liệu chấm công (`attendance.db`), môi trường ảo (`.venv/`), tệp mô hình nhị phân (`models/`) và tập ảnh thực tế của nhân viên (`data/faces/*`) đều được cấu hình loại trừ tuyệt đối trong [.gitignore](file:///d:/Do_an_AI/.gitignore).
  - Không bao giờ đưa dữ liệu định danh sinh trắc học cá nhân lên kho lưu trữ công cộng.
- **Thread & Hardware Safety**: Tự động dọn dẹp và giải phóng tài nguyên phần cứng camera ngay khi người dùng chuyển tab hoặc đóng trình duyệt, ngăn chặn hoàn toàn lỗi xung đột thiết bị.

---

## 📄 Bản Quyền & Giấy Phép (License)
Dự án được phát triển phục vụ mục đích nghiên cứu học thuật và đồ án giải pháp ứng dụng AI. Mọi quyền được bảo lưu theo giấy phép mã nguồn mở mở rộng.

