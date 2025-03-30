# 🧠 Smart Waste Classification & User Recognition API

An intelligent API system that combines **YOLOv11 for waste detection**, **YOLOv8 for face detection**, and **InsightFace for face recognition**, deployed with **Triton Inference Server** and served through a **FastAPI** backend. Ideal for smart environment monitoring, gamification, and community engagement.

---

## 🚀 Features

- 🗑️ Real-time **waste classification** (plastic, paper, metal...)
- 🧑‍💼 Face detection & recognition to **identify users**
- 📦 Register new users with `face_id` automatically
- 📊 Save logs: trash type, timestamp, user ID
- 🏆 Accumulate points for user participation
- 🌐 API-ready via FastAPI
- ⚡ ONNX model serving via **NVIDIA Triton Inference Server**
- ☁️ Cloud-native: Docker, Kubernetes, Helm supported

---

🏗️ **Project Structure**

```
.
├── app/                    # FastAPI app source code
│   ├── app.py              # API Endpoint
│   └── triton_clients/     # Triton gRPC clients for YOLO ONNX model
├── models/                 # ONNX model repository for Triton
├── Dockerfile              # Docker build config
├── docker-compose.yml      # Multi-container deployment (CPU/GPU)
├── requirements.txt        # Python dependencies
│   ├── triton_clients/     # Triton gRPC or HTTP clients for model inference
│   │   ├── yolo_client.py          # YOLO model inference client
│   │   ├── vae_encoder_client.py   # VAE Encoder client
│   │   ├── vae_decoder_client.py   # VAE Decoder client
│   │   ├── unet_client.py          # UNet client
├── charts/                 # Helm chart for Kubernetes deployment
│   └── waste-classifier/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           └── service.yaml
├── k8s/                    # Kubernetes manifests
│   ├── waste-classifier-deploy.yaml
│   └── waste-classifier-service.yaml
└── README.md
```

---

## 🔁 System Workflow

1. 📸 Camera captures trash disposal action
2. 🗑 **YOLOv11** detects and classifies trash
3. 🧑 **YOLOv8** detects face → **InsightFace** verifies user
4. 🧠 System checks:
   - If face is **known** → update log & increase points
   - If face is **new** → create user & save embedding
5. 📥 Record:
   - Trash type & count
   - Timestamp
   - `user_id`

---

## 🧾 Suggested Database Schema

### Table: `Users`

| Column         | Type      |
| -------------- | --------- |
| user_id        | UUID/Int  |
| face_embedding | BLOB      |
| points         | Integer   |
| registered_at  | Timestamp |

### Table: `TrashLogs`

| Column     | Type      |
| ---------- | --------- |
| id         | UUID/Int  |
| trash_type | String    |
| count      | Int       |
| timestamp  | Timestamp |
| user_id    | FK        |

---

## 🐳 Setup with Docker Triton Server

```bash
docker run --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 \
  -v /path/to/model:/models nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

---

## 🚀 Run FastAPI Server

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8080 --reload
```

---

## 🎯 Example Endpoint

```bash
POST /predict
Content-Type: multipart/form-data
Body:
- image: JPEG file
```

Returns:

```json
{
  "user_id": "abc123",
  "trash": [
    { "type": "plastic", "count": 2 },
    { "type": "paper", "count": 1 }
  ],
  "timestamp": "2025-03-30T10:12:00Z"
}
```

---

## 📦 Future Improvements

- Real-time dashboard (Streamlit / Grafana)
- Mobile-friendly QR code check-in
- Voice feedback / camera trigger
- Gamified leaderboard with rewards

---

## 📄 License

Open source for environmental AI and smart community systems.
