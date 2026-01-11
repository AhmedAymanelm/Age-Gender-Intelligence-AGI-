"""Configuration settings for the application."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "Gender and Age Detection API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Model paths
    models_dir: Path = Path("models")
    face_proto: Path = models_dir / "opencv_face_detector.pbtxt"
    face_model: Path = models_dir / "opencv_face_detector_uint8.pb"
    age_proto: Path = models_dir / "age_deploy.prototxt"
    age_model: Path = models_dir / "age_net.caffemodel"
    gender_proto: Path = models_dir / "gender_deploy.prototxt"
    gender_model: Path = models_dir / "gender_net.caffemodel"
    
    # Detection settings
    face_conf_threshold: float = 0.5
    padding: int = 20
    frames_to_stabilize: int = 3
    
    # Model parameters
    model_mean_values: tuple = (78.4263377603, 87.7689143744, 114.895847746)
    age_list: list = ['(0-2)', '(3-6)', '(7-12)', '(13-19)', '(20-29)',
                      '(30-39)', '(40-49)', '(50-59)', '(60-74)', '(75-100)']
    gender_list: list = ['Male', 'Female']
    
    # Directories
    faces_dir: Path = Path("facess")
    uploads_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    
    # Data storage
    data_file: Path = Path("detections.json")
    
    # Tracker settings
    max_age: int = 5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Create necessary directories
settings.faces_dir.mkdir(exist_ok=True)
settings.uploads_dir.mkdir(exist_ok=True)
settings.output_dir.mkdir(exist_ok=True)
