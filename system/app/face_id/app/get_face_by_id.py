from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse
from database import SessionLocal
from models import FaceProfile
import os

router = APIRouter()

@router.get("/faces/{face_id}")
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