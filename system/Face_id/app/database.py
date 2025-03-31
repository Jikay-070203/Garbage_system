import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from models import Base

# Cấu hình database
DB_USER = "root"
DB_PASSWORD = "070203"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "face_db"

# URL kết nối
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
DATABASE_URL_NO_DB = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/"

# Engine và Session
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

# Tạo database nếu chưa có
def create_database():
    engine_no_db = create_engine(DATABASE_URL_NO_DB, echo=True)
    with engine_no_db.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}"))
        print(f"✅ Database '{DB_NAME}' đã được tạo (nếu chưa có).")

# Tạo bảng
def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✅ Các bảng đã được tạo.")
