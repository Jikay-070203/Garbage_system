import os
from fastapi.staticfiles import StaticFiles

# --- Đường dẫn thư mục ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Thư mục chứa config.py
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # Truy về FASTAPI_V8

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
FACES_DB_DIR = os.path.join(PROJECT_DIR, "faces_db")
DB_DIR = os.path.join(PROJECT_DIR, "database")

# --- Tham số hệ thống ---
TRITON_URL = "localhost:8000"
IOU_THRESHOLD = 0.6
EMBEDDING_THRESHOLD = 0.5
COLORS = [(0, 255, 0)]

# --- Tạo thư mục nếu chưa tồn tại ---
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(FACES_DB_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)