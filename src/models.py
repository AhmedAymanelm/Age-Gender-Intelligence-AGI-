"""Data models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class DetectionResponse(BaseModel):
    """Response model for a single detection."""
    id: int
    image: str
    gender: str
    age: str
    entry_time: str


class VideoProcessRequest(BaseModel):
    """Request model for video processing."""
    video_path: str = Field(..., description="Path to video file or video URL")
    conf_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold for face detection")


class VideoProcessResponse(BaseModel):
    """Response model for video processing."""
    status: str
    message: str
    output_video: Optional[str] = None
    detections_count: int
    detections: list[DetectionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    app_name: str
    version: str
    models_loaded: bool


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
