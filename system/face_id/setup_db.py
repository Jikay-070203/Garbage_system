import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

# Cấu hình kết nối MySQL (giống database.py)
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "070203")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DB = os.getenv("MYSQL_DB", "face_db")

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
DATABASE_URL_NO_DB = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/"

def create_database():
    """Tạo database nếu chưa có (chỉ chạy 1 lần đầu)"""
    from sqlalchemy import text
    engine_no_db = create_engine(DATABASE_URL_NO_DB, echo=True)
    with engine_no_db.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
        print(f"✅ Database '{MYSQL_DB}' đã được tạo (nếu chưa có).")

def create_tables():
    engine = create_engine(DATABASE_URL, echo=True)
    Base.metadata.create_all(bind=engine)
    print("✅ Các bảng đã được tạo trên MySQL.")

def main():
    try:
        create_database()
    except Exception as e:
        print(f"(Có thể database đã tồn tại) {e}")
    create_tables()

if __name__ == "__main__":
    main() 