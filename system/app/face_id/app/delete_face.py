from fastapi import APIRouter
from database import SessionLocal, engine
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from config import FACES_DB_DIR
from database import SessionLocal
from models import FaceProfile, FaceProfileDeleted
import os, json
import cv2
import numpy as np
from sqlalchemy import text


router = APIRouter()
# Xóa toàn bộ
@router.delete("/faces/delete-all")
def delete_all_faces():
    session = SessionLocal()
    faces = session.query(FaceProfile).all()
    deleted = 0
    for face in faces:
        deleted_face = FaceProfileDeleted(
            face_id=face.face_id,
            timestamp=face.timestamp,
            image_path=face.image_path,
            embedding=face.embedding
        )
        session.add(deleted_face)
        session.delete(face)
        deleted += 1
    session.commit()
    session.close()
    return {"message": f"Deleted all {deleted} faces (moved to trash)"}

# Xóa nhiều gương mặt
@router.post("/faces/delete-multiple")
def delete_multiple_faces(face_ids: list[str] = Body(...)):
    session = SessionLocal()
    deleted = []
    for fid in face_ids:
        face = session.query(FaceProfile).filter(FaceProfile.face_id == fid).first()
        if face:
            deleted_face = FaceProfileDeleted(
                face_id=face.face_id,
                timestamp=face.timestamp,
                image_path=face.image_path,
                embedding=face.embedding
            )
            session.add(deleted_face)
            session.delete(face)
            deleted.append(fid)
    session.commit()
    session.close()
    return {"message": "Deleted successfully", "deleted": deleted}


# Xóa 1 gương mặt (chuyển sang bảng deleted)
@router.delete("/faces/{face_id}")
def delete_face_by_id(face_id: str):
    session = SessionLocal()
    face = session.query(FaceProfile).filter(FaceProfile.face_id == face_id).first()
    if face:
        deleted_face = FaceProfileDeleted(
            face_id=face.face_id,
            timestamp=face.timestamp,
            image_path=face.image_path,
            embedding=face.embedding
        )
        session.add(deleted_face)
        session.delete(face)
        session.commit()
        session.close()
        return JSONResponse(content={"message": "Moved to trash (deleted) successfully"})
    session.close()
    return JSONResponse(content={"error": "Face not found"}, status_code=404)

# Phục hồi 1 gương mặt từ bảng deleted
@router.post("/faces/restore/{face_id}")
def restore_face(face_id: str):
    session = SessionLocal()
    face = session.query(FaceProfileDeleted).filter(FaceProfileDeleted.face_id == face_id).first()
    if face:
        restored_face = FaceProfile(
            face_id=face.face_id,
            timestamp=face.timestamp,
            image_path=face.image_path,
            embedding=face.embedding
        )
        session.add(restored_face)
        session.delete(face)
        session.commit()
        session.close()
        return {"message": "Restored successfully"}
    session.close()
    return {"error": "Face not found in trash"}

#phục hồi tùy chọn 
@router.post("/faces/restore-multiple")
def restore_multiple_faces(face_ids: list[str] = Body(...)):
    session = SessionLocal()
    restored = []
    for fid in face_ids:
        face = session.query(FaceProfileDeleted).filter(FaceProfileDeleted.face_id == fid).first()
        if face:
            restored_face = FaceProfile(
                face_id=face.face_id,
                timestamp=face.timestamp,
                image_path=face.image_path,
                embedding=face.embedding
            )
            session.add(restored_face)
            session.delete(face)
            restored.append(fid)
    session.commit()
    session.close()
    return {
        "message": f"Restored {len(restored)} faces successfully",
        "restored": restored
    }
    
# Phục hồi toàn bộ từ bảng deleted
@router.post("/faces/restore-all")
def restore_all_faces():
    session = SessionLocal()
    deleted_faces = session.query(FaceProfileDeleted).all()

    if not deleted_faces:
        session.close()
        return {"message": "Không có gương mặt nào trong thùng rác."}

    existing_ids = set([f.id for f in session.query(FaceProfile.id).all()])
    restored_count = 0

    for face in deleted_faces:
        if face.id in existing_ids:
            continue  # tránh id trùng gây lỗi
        restored_face = FaceProfile(
            id=face.id,  # giữ nguyên ID gốc
            face_id=face.face_id,
            timestamp=face.timestamp,
            image_path=face.image_path,
            embedding=face.embedding
        )
        session.add(restored_face)
        session.delete(face)
        restored_count += 1

    session.commit()

    # Cập nhật lại AUTO_INCREMENT cho bảng face_profiles
    max_id = session.query(FaceProfile).order_by(FaceProfile.id.desc()).first()
    if max_id:
        session.execute(text(f"ALTER TABLE face_profiles AUTO_INCREMENT = {max_id.id + 1}"))
        session.commit()

    session.close()
    return {"message": f"Đã phục hồi {restored_count} gương mặt và cập nhật lại AUTO_INCREMENT"}

# Nhận diện realtime
def run_realtime_inference(face_embed_model, EMBEDDING_THRESHOLD):
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
