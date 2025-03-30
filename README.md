# 🧠 Smart Waste Classification & User Recognition API

## Waste detection system, facial detection and recognition, deployed with Triton Inference Server and served via FastAPI backend. Ideal for smart environmental monitoring, community interaction.

## 🚀 Features

- 🗑️ Real-time **waste classification** (plastic, paper, metal...)
- 🧑‍💼 Face detection & recognition to **identify users**
- 📦 Register new users with `face_id` automatically
- 📊 Save logs: trash type, timestamp, user ID
- 🌐 API-ready via FastAPI
- ⚡ ONNX model serving via **NVIDIA Triton Inference Server**
- ☁️ Cloud-native: Docker - (Kubernetes, Helm supported)

* Add function in futute

- (🏆 Accumulate points for user participation)
- (Kubernetes, Helm supported)

---

🏗️ **Project Structure**

```
📁 Project Root
├── 📁 app
├── 📁 data
├── 📁 doc
├── 📁 information
├── 📁 models
├── 📁 output
├── 📁 server
├── 📁 up
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 thông tin hệ thống.docx

```

---

## 🔁 System Workflow

1. 📸 Camera captures trash disposal action
2. 🗑 detects and classifies trash
3. 🧑detects face → verifies user
4. 🧠 System checks:
   - If face is **known** → update log & increase points
   - If face is **new** → create user & save embedding
5. 📥 Record:
   - Trash type & count
   - Timestamp
   - `user_id`

## 🐳 Setup with Docker Triton Server

```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/model:/models nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

---

## 🚀 Run FastAPI Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

## 📦 Future Improvements

- Real-time dashboard (Streamlit / Grafana)
- Mobile-friendly QR code check-in
- Voice feedback / camera trigger
- Gamified leaderboard with rewards

---
