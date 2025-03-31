import datetime
from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# 1. Hồ sơ khuôn mặt
class FaceProfile(Base):
    __tablename__ = "face_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String(255))
    embedding = Column(Text)

    # Quan hệ với bảng người dùng
    user = relationship("User", uselist=False, back_populates="face_profile")

# 2. Hồ sơ khuôn mặt đã xóa (chờ phục hồi)
class FaceProfileDeleted(Base):
    __tablename__ = "face_profiles_deleted"
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_id = Column(String(100), unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String(255))
    embedding = Column(Text)

# 3. Thông tin người dùng
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100))
    email = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    points = Column(Integer, default=0)

    face_id = Column(String(100), ForeignKey("face_profiles.face_id"), unique=True)
    face_profile = relationship("FaceProfile", back_populates="user")

# 4. Danh sách loại rác
class GarbageType(Base):
    __tablename__ = "garbage_types"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True)
    description = Column(Text, nullable=True)
    point_value = Column(Integer, default=0)  # điểm cộng cho mỗi loại rác

# 5. Lượt bỏ rác của người dùng
class GarbageEntry(Base):
    __tablename__ = "garbage_entries"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    garbage_type_id = Column(Integer, ForeignKey("garbage_types.id"))
    timestamp = Column(DateTime, default=datetime.datetime.now)
    image_path = Column(String(255))
    location = Column(String(255), nullable=True)

    user = relationship("User", backref="garbage_entries")
    garbage_type = relationship("GarbageType")

# 6. Lịch sử điểm
class PointsHistory(Base):
    __tablename__ = "points_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    change_type = Column(String(50))  # 'add', 'deduct', 'bonus'
    points = Column(Integer)
    description = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.now)

    user = relationship("User", backref="points_history")



# class FaceProfile(Base):
#     __tablename__ = "face_profiles"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     face_id = Column(String(100), unique=True, nullable=False)
#     timestamp = Column(DateTime, default=datetime.datetime.now)
#     image_path = Column(String(255))
#     embedding = Column(Text)
    
#     #database xóa tạm chờ restore 
# class FaceProfileDeleted(Base):
#     __tablename__ = "face_profiles_deleted"
#     id = Column(Integer, primary_key=True, autoincrement=True)
#     face_id = Column(String(100), unique=True, nullable=False)
#     timestamp = Column(DateTime, default=datetime.datetime.now)
#     image_path = Column(String(255))
#     embedding = Column(Text)
