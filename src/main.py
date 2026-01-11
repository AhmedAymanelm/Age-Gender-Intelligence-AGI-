"""FastAPI application for Gender and Age Detection."""
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import json
from typing import List

from .config import settings
from .models import (
    DetectionResponse,
    VideoProcessRequest,
    VideoProcessResponse,
    HealthResponse,
    ErrorResponse
)
from .detector import get_detector
from .processor import VideoProcessor

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API for detecting gender and age from video streams"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
video_processor = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    try:
        # Load detector (this will load all models)
        detector = get_detector()
        print(f"✓ {settings.app_name} started successfully")
    except Exception as e:
        print(f"✗ Failed to start application: {str(e)}")
        raise


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - Health check."""
    try:
        detector = get_detector()
        models_loaded = True
    except:
        models_loaded = False
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        app_name=settings.app_name,
        version=settings.app_version,
        models_loaded=models_loaded
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        detector = get_detector()
        models_loaded = True
    except:
        models_loaded = False
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        app_name=settings.app_name,
        version=settings.app_version,
        models_loaded=models_loaded
    )


@app.post("/process-video", response_model=VideoProcessResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
):
    """
    Process uploaded video for face detection and age/gender prediction.
    
    Args:
        video: Video file to process
        
    Returns:
        VideoProcessResponse with results
    """
    try:
        # Save uploaded video
        video_path = settings.uploads_dir / video.filename
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video
        processor = VideoProcessor()
        result = processor.process_video(str(video_path))
        
        # Clean up uploaded file in background
        background_tasks.add_task(cleanup_file, video_path)
        
        return VideoProcessResponse(
            status="success",
            message=f"Video processed successfully. Detected {result['detections_count']} unique persons.",
            output_video=result["output_video"],
            detections_count=result["detections_count"],
            detections=result["detections"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-video-path", response_model=VideoProcessResponse)
async def process_video_from_path(request: VideoProcessRequest):
    """
    Process video from file path.
    
    Args:
        request: VideoProcessRequest with video path
        
    Returns:
        VideoProcessResponse with results
    """
    try:
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video file not found")
        
        # Process video
        processor = VideoProcessor()
        result = processor.process_video(str(video_path))
        
        return VideoProcessResponse(
            status="success",
            message=f"Video processed successfully. Detected {result['detections_count']} unique persons.",
            output_video=result["output_video"],
            detections_count=result["detections_count"],
            detections=result["detections"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detections", response_model=List[DetectionResponse])
async def get_detections():
    """Get all detections from database."""
    try:
        if not settings.data_file.exists():
            return []
        
        with open(settings.data_file, "r") as f:
            data = json.load(f)
        
        return data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/detections/{detection_id}", response_model=DetectionResponse)
async def get_detection(detection_id: int):
    """Get specific detection by ID."""
    try:
        if not settings.data_file.exists():
            raise HTTPException(status_code=404, detail="No detections found")
        
        with open(settings.data_file, "r") as f:
            data = json.load(f)
        
        detection = next((d for d in data if d["id"] == detection_id), None)
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        return detection
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download-output/{filename}")
async def download_output(filename: str):
    """Download output video."""
    try:
        file_path = settings.output_dir / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(
            path=str(file_path),
            media_type="video/mp4",
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/face-image/{person_id}")
async def get_face_image(person_id: int):
    """Get face image for a person."""
    try:
        image_path = settings.faces_dir / f"person_{person_id}.jpg"
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")
        
        return FileResponse(
            path=str(image_path),
            media_type="image/jpeg"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/detections")
async def clear_detections():
    """Clear all detections."""
    try:
        # Clear JSON file
        if settings.data_file.exists():
            with open(settings.data_file, "w") as f:
                json.dump([], f)
        
        # Clear face images
        for image in settings.faces_dir.glob("*.jpg"):
            image.unlink()
        
        return {"status": "success", "message": "All detections cleared"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def cleanup_file(file_path: Path):
    """Clean up temporary file."""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
