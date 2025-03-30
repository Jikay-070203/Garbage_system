from fastapi import FastAPI
from list_faces import router as faces_router
from get_face_by_id import router as face_detail_router
from delete_face import router as delete_router
from predict import router as predict_router
from config import FACES_DB_DIR
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from config import MODEL_DIR
from face_model import face_embed_model
from config import MODEL_DIR, FACES_DB_DIR, DB_DIR
from models import Base
from database import engine

################
import os
import insightface
from insightface.model_zoo import model_zoo
from database import create_tables
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from fastapi import Query
from starlette.responses import Response
import pandas as pd
from fastapi.staticfiles import StaticFiles


app = FastAPI()

app.include_router(predict_router)
app.include_router(faces_router)
app.include_router(face_detail_router)
app.include_router(delete_router)

Base.metadata.create_all(bind=engine)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # Vị trí file app.py
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # Thư mục FASTAPI_V8
MODEL_DIR = os.path.join(PROJECT_DIR, "models")             # Tạo đường dẫn models/buffalo_l
model_path = os.path.join(MODEL_DIR, "buffalo_l")

# === Tạo thư mục nếu chưa có ===
os.makedirs(model_path, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "buffalo_l")
if not os.path.exists(model_path):
    print("🔄 Đang tải mô hình từ Hugging Face...")
    snapshot_download(repo_id="hoanguyenthanh07/buffalo_l", local_dir=model_path)
    print("✅ Tải xuống thành công!")

face_embed_model = FaceAnalysis(name="buffalo_l", root=model_path, providers=["CPUExecutionProvider"])
face_embed_model.prepare(ctx_id=-1, det_size=(640, 640))

app.mount("/faces_db", StaticFiles(directory=FACES_DB_DIR), name="faces_db")

