from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base

# Cấu hình thông số kết nối
DB_USER = "root"
DB_PASSWORD = "070203"
DB_HOST = "localhost"
DB_PORT = "3306"
DB_NAME = "face_db"

# Tạo URL kết nối
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Khởi tạo engine và session
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)

def create_tables():
    Base.metadata.create_all(bind=engine)
    
    
