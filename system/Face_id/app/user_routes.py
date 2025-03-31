from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User, FaceProfile, GarbageEntry, GarbageType
import schemas
from typing import List

router = APIRouter()

# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/user")
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    face = db.query(FaceProfile).filter(FaceProfile.face_id == user.face_id).first()
    if not face:
        raise HTTPException(status_code=404, detail="Face profile not found.")
    if db.query(User).filter(User.face_id == user.face_id).first():
        raise HTTPException(status_code=400, detail="User with this face_id already exists.")
    
    new_user = User(**user.dict())
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"message": "User created successfully", "user_id": new_user.id}

@router.post("/entry")
def create_entry(entry: schemas.GarbageEntryCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.face_id == entry.face_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    
    garbage_type = db.query(GarbageType).filter(GarbageType.id == entry.garbage_type_id).first()
    if not garbage_type:
        raise HTTPException(status_code=404, detail="Garbage type not found.")
    
    new_entry = GarbageEntry(
        user_id=user.id,
        garbage_type_id=entry.garbage_type_id,
        image_path=entry.image_path,
        location=entry.location
    )
    db.add(new_entry)

    user.points += garbage_type.point_value

    db.commit()
    return {"message": "Garbage entry recorded & points updated."}

@router.get("/leaderboard", response_model=list[schemas.LeaderboardItem])
def get_leaderboard(db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.points.desc()).limit(10).all()
    return [{"name": u.name, "points": u.points} for u in users]


# Lấy danh sách người dùng
@router.get("/users", response_model=List[schemas.UserInfo])
def get_all_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

# Lấy thông tin người dùng theo face_id
@router.get("/user/{face_id}", response_model=schemas.UserInfo)
def get_user_by_face_id(face_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.face_id == face_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    return user