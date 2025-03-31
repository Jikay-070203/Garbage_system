from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    face_id: str
    
class UserInfo(BaseModel):
    id: int
    name: str
    email: Optional[str]
    phone: Optional[str]
    face_id: str
    points: int

    class Config:
        orm_mode = True

class GarbageEntryCreate(BaseModel):
    face_id: str
    garbage_type_id: int
    image_path: Optional[str] = None
    location: Optional[str] = None

class LeaderboardItem(BaseModel):
    name: str
    points: int
