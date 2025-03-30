from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse, HTMLResponse
from database import SessionLocal, create_tables
from models import FaceProfile
import tempfile
import json
import os

router = APIRouter()

@router.get("/faces/")
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