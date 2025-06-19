CREATE DATABASE face_db;
USE face_db;

CREATE TABLE face_profiles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    face_id VARCHAR(255) NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    image_path VARCHAR(500) NOT NULL,
    embedding LONGTEXT NOT NULL -- Lưu dưới dạng JSON string
);

