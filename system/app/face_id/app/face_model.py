import os
from insightface.app import FaceAnalysis
from huggingface_hub import snapshot_download
from config import MODEL_DIR  # 👈 import đúng đường dẫn từ config

MODEL_PATH = os.path.join(MODEL_DIR, "buffalo_l")

# Tạo thư mục nếu chưa có
os.makedirs(MODEL_PATH, exist_ok=True)

# Tải model nếu chưa có
if not os.listdir(MODEL_PATH):  # Chỉ tải nếu thư mục trống
    print("🔄 Đang tải mô hình từ Hugging Face...")
    snapshot_download(
        repo_id="hoanguyenthanh07/buffalo_l",
        local_dir=MODEL_PATH,
        local_dir_use_symlinks=False
    )
    print("✅ Tải mô hình thành công!")

# Khởi tạo model
face_embed_model = FaceAnalysis(name="buffalo_l", root=MODEL_PATH, providers=["CPUExecutionProvider"])
face_embed_model.prepare(ctx_id=-1, det_size=(640, 640))
