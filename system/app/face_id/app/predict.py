from fastapi import APIRouter, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import cv2
import json
import datetime
import os
import io
from PIL import Image
import tritonclient.http as httpclient
from database import SessionLocal
from models import FaceProfile
from predict_utils import xywh2xyxy, auto_conf_threshold, estimate_target_box_count, is_new_face
from face_model import face_embed_model 

router = APIRouter()

client = httpclient.InferenceServerClient(url="localhost:8000")
IOU_THRESHOLD = 0.6
COLORS = [(0, 255, 0)]
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "faces_db")

@router.post("/predict/", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...), EMBEDDING_THRESHOLD: float = 0.5):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_img = np.array(image)

    img = cv2.resize(orig_img, (640, 640))
    img_input = img.transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)

    inputs = [httpclient.InferInput("images", img_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(img_input)
    outputs = [httpclient.InferRequestedOutput("output0")]
    response = client.infer("Onnx", inputs=inputs, outputs=outputs)
    pred = np.squeeze(response.as_numpy("output0")).T

    target_box_count = estimate_target_box_count(pred, min_conf=0.15)
    CONF_THRESHOLD = auto_conf_threshold(pred, target_box_count)

    boxes, scores = [], []
    for i in range(pred.shape[0]):
        x, y, w, h, conf = pred[i][:5]
        if conf > CONF_THRESHOLD:
            box = xywh2xyxy([
                x * orig_img.shape[1],
                y * orig_img.shape[0],
                w * orig_img.shape[1],
                h * orig_img.shape[0]
            ])
            boxes.append([int(b) for b in box])
            scores.append(float(conf))

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
    print(f"📸 Số khuôn mặt phát hiện sau NMS: {len(indices)}")

    detected_faces = face_embed_model.get(orig_img)
    print(f"🔍 Mô hình nhận diện được: {len(detected_faces)} khuôn mặt")

    db_session = SessionLocal()
    existing_embeddings = [
        np.array(json.loads(p.embedding)) for p in db_session.query(FaceProfile).all()
    ]

    for face in detected_faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embed = face.embedding

        if is_new_face(embed, existing_embeddings, EMBEDDING_THRESHOLD):
            face_id = f"face_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            path = os.path.join(DB_DIR, f"{face_id}.jpg")
            face_crop = orig_img[y1:y2, x1:x2]
            face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, face_bgr)
            try:
                new_face = FaceProfile(
                    face_id=face_id,
                    timestamp=datetime.datetime.now(),
                    image_path=path,
                    embedding=json.dumps(embed.tolist())
                )
                db_session.add(new_face)
                db_session.commit()
                existing_embeddings.append(embed)
                label = f"New Face"
                print(f"✅ Lưu khuôn mặt mới vào DB: {face_id}")
            except Exception as e:
                print(f"❌ Lỗi lưu DB: {e}")
                label = f"Error"
        else:
            label = f"Known Face"

        cv2.rectangle(orig_img, (x1, y1), (x2, y2), COLORS[0], 2)
        cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

    db_session.close()

    cv2.putText(orig_img, f"Face count: {len(detected_faces)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    print(f"🧠 Số khuôn mặt được InsightFace nhận diện: {len(detected_faces)}")

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", orig_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

