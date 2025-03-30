from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base
import datetime
from sqlalchemy import Column, Boolean

Base = declarative_base()

class FaceProfile(Base):
    __tablename__ = "face_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String(255))
    embedding = Column(Text)
    
    #database xóa tạm chờ restore 
class FaceProfileDeleted(Base):
    __tablename__ = "face_profiles_deleted"
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String(255))
    embedding = Column(Text)
