# ♻️ Smart Waste Classification & User Recognition System

## Tổng quan

Hệ thống phân loại rác thông minh kết hợp nhận diện người dùng qua khuôn mặt, phục vụ giám sát môi trường, tích điểm , và quản lý dữ liệu qua API. Hệ thống sử dụng các mô hình AI (YOLO, InsightFace) triển khai trên Triton Inference Server, giao tiếp qua FastAPI, hỗ trợ Docker, dễ dàng mở rộng cho các ứng dụng thực tế tại trường học và cộng đồng.

---

## Tính năng nổi bật

- 🚮 **Phân loại rác** (nhựa, giấy, kim loại, ...), nhận diện qua ảnh/video/thực tế
- 🧑‍💼 **Nhận diện khuôn mặt**: xác định, đăng ký, quản lý người dùng
- 🆔 **Tự động tạo user** khi phát hiện khuôn mặt mới
- 🗂️ **Lưu log**: loại rác, thời gian, user, điểm tích lũy
- ⚡ **API chuẩn REST** với FastAPI
- 🖥️ **Triển khai mô hình ONNX qua NVIDIA Triton Inference Server**
- 🐳 **Hỗ trợ Docker, cloud-native** (có thể mở rộng Kubernetes, Helm)
- 🏆 **Tích điểm, bảng xếp hạng** cho người dùng

---

## Cấu trúc thư mục

```
Garbage_system/
├── README.md
├── system/
│   ├── Face_id/         # Nhận diện khuôn mặt, quản lý user
│   ├── Garbage/         # Phân loại rác, mô hình, API
│   ├── information/     # Quản lý thông tin, cập nhật dữ liệu
│   ├── server/          # Script, cấu hình Triton, Docker
│   ├── requirements.txt # Thư viện Python cần thiết
│   └── ...
└── ...
```

---

## Yêu cầu hệ thống

- Python >= 3.8
- CUDA (nếu dùng GPU)
- Docker (nếu dùng Triton)
- Các thư viện: ultralytics, fastapi, uvicorn, tritonclient[all], opencv-python, torch, insightface, sqlalchemy, pymysql, pandas, ...

---

## Hướng dẫn cài đặt

### 1. Cài đặt Python & thư viện

```bash
pip install -r system/requirements.txt
```

### 2. Chạy Triton Inference Server (nếu dùng mô hình ONNX)

```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/model:/models nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### 3. Chạy FastAPI server (ví dụ cho module Face_id)

```bash
cd system/Face_id/app
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### 4. Chạy API phân loại rác (Garbage)

```bash
cd system/Garbage/app
uvicorn appapi:app --host 0.0.0.0 --port 8000 --reload
```

---

## Sơ đồ luồng hệ thống

1. 📷 Camera ghi nhận hành động bỏ rác
2. 🚮 Nhận diện, phân loại rác
3. 🧑 Nhận diện khuôn mặt → xác định user
4. 🏅 Nếu user đã có: cập nhật log, cộng điểm
5. 🆕 Nếu user mới: tạo user, lưu embedding
6. 🗃️ Ghi nhận: loại rác, thời gian, user_id, điểm

---

## API chính

### Nhận diện rác

- `POST /detect/` : Nhận ảnh, trả về ảnh có bounding box, tên loại rác, độ tin cậy
- `GET /status/` : Kiểm tra trạng thái model

### Nhận diện khuôn mặt & quản lý user

- `POST /user` : Đăng ký user mới (gắn với face_id)
- `POST /entry` : Ghi nhận hành động bỏ rác, cộng điểm
- `GET /leaderboard` : Lấy top user tích điểm
- `GET /users` : Lấy danh sách user
- `GET /user/{face_id}` : Lấy thông tin user theo face_id

### (Tham khảo thêm chi tiết trong code từng module)

---

## Phát triển tương lai

- Dashboard realtime (Streamlit/Grafana)
- Check-in QR code, mobile app
- Voice feedback, camera trigger
- Gamification, phần thưởng
- Hỗ trợ Kubernetes, Helm

---

## CUSTOM LICENSE

Copyright (c) 2025 Nguyen Thanh Hoa

Bạn được phép sử dụng, sao chép, chỉnh sửa, chia sẻ, thương mại hóa dự án này với điều kiện:

- Ghi rõ nguồn tác giả: Nguyen Thanh Hoa
- Không được kiện cáo, phản bác hay gây ảnh hưởng tiêu cực đến tác giả
- Mọi thay đổi phải nêu rõ và giữ lại phần ghi công gốc

Dự án được cung cấp "nguyên trạng", không có bất kỳ bảo đảm nào. Tác giả không chịu trách nhiệm cho mọi rủi ro phát sinh từ việc sử dụng.
