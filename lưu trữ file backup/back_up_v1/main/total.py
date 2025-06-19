from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import tritonclient.http as httpclient
from PIL import Image
import io

app = FastAPI()
triton_url = "localhost:8000"
client = httpclient.InferenceServerClient(url=triton_url)

IOU_THRESHOLD = 0.6
CLASSES = ['face']
COLORS = [(0, 255, 0)]

def xywh2xyxy(x):
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y

def auto_conf_threshold(pred, target_box_count=3, min_conf=0.1, max_conf=0.9, step=0.01):
    thresholds = np.arange(max_conf, min_conf - step, -step)
    for threshold in thresholds:
        count = np.sum(pred[:, 4] >= threshold)
        if count >= target_box_count:
            return float(threshold)
    return min_conf

def estimate_target_box_count(pred, min_conf=0.1):
    return int(np.sum(pred[:, 4] >= min_conf))

@app.post("/predict/", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_img = np.array(image)
    img = cv2.resize(orig_img, (640, 640))
    img_input = img.transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)

    # Triton inference
    inputs = [httpclient.InferInput("images", img_input.shape, "FP32")]
    inputs[0].set_data_from_numpy(img_input)
    outputs = [httpclient.InferRequestedOutput("output0")]
    response = client.infer("Onnx", inputs=inputs, outputs=outputs)
    pred = np.squeeze(response.as_numpy("output0")).T  # (8400, 5)

    # 🧠 Ước lượng và tự chọn CONF_THRESHOLD
    target_box_count = estimate_target_box_count(pred, min_conf=0.15)
    CONF_THRESHOLD = auto_conf_threshold(pred, target_box_count)
    
    boxes, scores, class_ids = [], [], []
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
            class_ids.append(0)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        label = f"{CLASSES[class_ids[i]]}: {scores[i]:.2f}"
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), COLORS[0], 2)
        cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)

    # Trả ảnh trực tiếp về Swagger
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", orig_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
############################################
############################################
############################################
############################################
#V2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import cv2
import tritonclient.http as httpclient
from PIL import Image
import io
import os
import datetime
import json
import insightface
from insightface.model_zoo import model_zoo
from models import FaceProfile
from database import SessionLocal, create_tables
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from fastapi.responses import PlainTextResponse, HTMLResponse
from fastapi import Query
from fastapi.responses import PlainTextResponse, HTMLResponse, JSONResponse, FileResponse
from starlette.responses import Response
import tempfile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, PlainTextResponse
import pandas as pd

app = FastAPI()

# Xác định đường dẫn thư mục lưu dữ liệu
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_DIR = os.path.join(BASE_DIR, "database")  # Sửa lại tên thư mục nếu cần
FACES_DB_DIR = os.path.join(BASE_DIR, "faces_db")

# --- Khởi tạo thư mục lưu ảnh nếu chưa có ---
os.makedirs(DB_DIR, exist_ok=True)  # Bây giờ DB_DIR đã được định nghĩa trước khi sử dụng
os.makedirs(FACES_DB_DIR, exist_ok=True)  # Đảm bảo thư mục faces_db tồn tại

# Gắn thư mục vào FastAPI
app.mount("/faces_db", StaticFiles(directory=FACES_DB_DIR), name="faces_db")

triton_url = "localhost:8000"
client = httpclient.InferenceServerClient(url=triton_url)

IOU_THRESHOLD = 0.6
CLASSES = ['face']
COLORS = [(0, 255, 0)]

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FACES_DB_DIR = os.path.join(BASE_DIR, "faces_db")

EMBEDDING_THRESHOLD = 0.5  # cosine distance threshold

# --- Tạo bảng MySQL nếu chưa có ---
create_tables()

# --- Khởi tạo model nhận diện từ InsightFace từ thư mục local ---
model_dir = r"D:\\SourceCode\\ProjectOJT\\complete\\OJT_TASK3_LOCAL\\FASTAPI_V8\\models"
model_path = os.path.join(model_dir, "buffalo_l")

try:
    if not os.path.exists(model_path):
        print("🔄 Đang tải mô hình từ Hugging Face...")
        os.makedirs(model_dir, exist_ok=True)
        snapshot_download(repo_id="hoanguyenthanh07/buffalo_l", local_dir=model_path)
        print("✅ Tải xuống thành công!")

    face_embed_model = FaceAnalysis(name="buffalo_l", root=model_path, providers=["CPUExecutionProvider"])
    face_embed_model.prepare(ctx_id=-1, det_size=(640, 640))

    print("✅ Mô hình InsightFace đã khởi động thành công từ thư mục đã tải về!")

except Exception as e:
    raise RuntimeError(f"❌ Lỗi khi khởi tạo mô hình InsightFace: {e}")

def xywh2xyxy(x):
    y = np.copy(x)
    y[0] = x[0] - x[2] / 2
    y[1] = x[1] - x[3] / 2
    y[2] = x[0] + x[2] / 2
    y[3] = x[1] + x[3] / 2
    return y

def auto_conf_threshold(pred, target_box_count=3, min_conf=0.1, max_conf=0.9, step=0.01):
    thresholds = np.arange(max_conf, min_conf - step, -step)
    for threshold in thresholds:
        count = np.sum(pred[:, 4] >= threshold)
        if count >= target_box_count:
            return float(threshold)
    return min_conf

def estimate_target_box_count(pred, min_conf=0.1):
    return int(np.sum(pred[:, 4] >= min_conf))

def is_new_face(embedding, db_embeddings, threshold=EMBEDDING_THRESHOLD):
    for known_embed in db_embeddings:
        sim = np.dot(embedding, known_embed) / (np.linalg.norm(embedding) * np.linalg.norm(known_embed))
        if sim > (1 - threshold):
            return False
    return True

@app.post("/predict/", response_class=StreamingResponse)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_img = np.array(image)

    img = cv2.resize(orig_img, (640, 640))
    img_input = img.transpose(2, 0, 1) / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)

    # Triton inference
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

    # Nhận diện từ ảnh gốc luôn
    detected_faces = face_embed_model.get(orig_img)
    print(f"🔍 Mô hình nhận diện được: {len(detected_faces)} khuôn mặt")

    db_session = SessionLocal()
    existing_embeddings = [
        np.array(json.loads(p.embedding)) for p in db_session.query(FaceProfile).all()
    ]

    for face in detected_faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embed = face.embedding

        if is_new_face(embed, existing_embeddings):
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

    cv2.putText(orig_img, f"Face count: {len(indices)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", orig_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")

from fastapi import Query
from fastapi.responses import PlainTextResponse, HTMLResponse

@app.get("/faces/")
def list_faces(format: str = Query("json", enum=["json", "csv", "html", "json_file", "csv_file", "html_file"])):
    session = SessionLocal()
    faces = session.query(FaceProfile).all()
    session.close()

    # Dữ liệu chung
    result = [{
        "id": f.id,
        "face_id": f.face_id,
        "timestamp": f.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": f.image_path,
        "embedding": json.loads(f.embedding)
    } for f in faces]

    # --- 1. JSON API Response ---
    if format == "json":
        return JSONResponse(content=result)

    # --- 2. JSON File Download ---
    if format == "json_file":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode='w', encoding='utf-8')
        json.dump(result, tmp, ensure_ascii=False, indent=2)
        tmp.close()
        return FileResponse(tmp.name, filename="faces.json", media_type="application/json")

    # --- 3. CSV ---
    csv_content = "id,face_id,timestamp,image_path,embedding\n"
    for f in result:
        csv_content += f'{f["id"]},{f["face_id"]},{f["timestamp"]},{f["image_path"]},"{json.dumps(f["embedding"])}"\n'

    if format == "csv":
        return PlainTextResponse(content=csv_content, media_type="text/csv")

    if format == "csv_file":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w', encoding='utf-8')
        tmp.write(csv_content)
        tmp.close()
        return FileResponse(tmp.name, filename="faces.csv", media_type="text/csv")

    # --- 4. HTML ---
    html = """
    <html><head><title>Face Database</title></head><body>
    <h2>DANH SÁCH FACE</h2>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>ID</th>
            <th>Face ID</th>
            <th>Timestamp</th>
            <th>Image</th>
            <th>Embedding (preview)</th>
        </tr>
    """

    for f in result:
        embed_preview = ", ".join(map(str, f["embedding"][:5]))  # 5 giá trị đầu
        image_src = "/" + f["image_path"].replace("\\", "/")     # ✅ sửa tại đây
        img_tag = f'<img src="{image_src}" width="100" />' if os.path.exists(f["image_path"]) else "(No image)"
        html += f"""
        <tr>
            <td>{f["id"]}</td>
            <td>{f["face_id"]}</td>
            <td>{f["timestamp"]}</td>
            <td>{img_tag}</td>
            <td>{embed_preview}...</td>
        </tr>
        """
    html += "</table></body></html>"

    if format == "html":
        return HTMLResponse(content=html)

    if format == "html_file":
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode='w', encoding='utf-8')
        tmp.write(html)
        tmp.close()
        return FileResponse(tmp.name, filename="faces.html", media_type="text/html")

@app.get("/faces/{face_id}")
def get_face_by_id(face_id: str):
    session = SessionLocal()
    face = session.query(FaceProfile).filter(FaceProfile.face_id == face_id).first()
    session.close()
    
    if face:
        image_url = "/" + face.image_path.replace("\\", "/")  # đường dẫn tĩnh đến ảnh
        html_content = f"""
        <html>
            <head><title>Face Detail</title></head>
            <body>
                <h2>Thông tin khuôn mặt</h2>
                <p><strong>Face ID:</strong> {face.face_id}</p>
                <p><strong>Timestamp:</strong> {face.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Hình ảnh:</strong><br>
                    <img src="{image_url}" width="200" />
                </p>
                <p><a href="{image_url}" download>📥 Tải ảnh</a></p>
            </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    return JSONResponse(content={"error": "Face not found"}, status_code=404)

@app.delete("/faces/{face_id}")
def delete_face_by_id(face_id: str):
    session = SessionLocal()
    face = session.query(FaceProfile).filter(FaceProfile.face_id == face_id).first()
    if face:
        if os.path.exists(face.image_path):
            os.remove(face.image_path)
        session.delete(face)
        session.commit()
        session.close()
        return JSONResponse(content={"message": "Deleted successfully"})
    session.close()
    return JSONResponse(content={"error": "Face not found"}, status_code=404)

def run_realtime_inference():
    cap = cv2.VideoCapture(0)
    db_session = SessionLocal()
    known_faces = db_session.query(FaceProfile).all()
    embeddings = [(f.face_id, np.array(json.loads(f.embedding))) for f in known_faces]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_embed_model.get(frame)
        for face in faces:
            embed = face.embedding
            name = "Unknown"
            for fid, known_embed in embeddings:
                sim = np.dot(embed, known_embed) / (np.linalg.norm(embed) * np.linalg.norm(known_embed))
                if sim > (1 - EMBEDDING_THRESHOLD):
                    name = fid
                    break

            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("Realtime Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    db_session.close()
