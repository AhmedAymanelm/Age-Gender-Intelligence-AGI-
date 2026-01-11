# Age & Gender Intelligence (AGI)

FastAPI application for detecting gender and age from video streams using deep learning models.

## Features

-  Video processing with face detection
- Age and gender prediction
-  Person tracking using DeepSORT
-  JSON data storage
-  Face image extraction
- RESTful API with FastAPI

## Project Structure

```
gender and age/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration settings
│   ├── detector.py          # Face detection and prediction
│   ├── processor.py         # Video processing logic
│   ├── models.py            # Pydantic data models
│   └── main.py              # FastAPI application
├── models/                  # Deep learning models (place model files here)
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── age_deploy.prototxt
│   ├── age_net.caffemodel
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
├── facess/                  # Extracted face images
├── uploads/                 # Temporary uploaded videos
├── outputs/                 # Processed output videos
├── requirements.txt         # Python dependencies
├── run.py                   # Application runner
├── .env.example             # Environment variables example
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Installation

### 1. Clone or Download the Project

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Model Files

Place the required model files in the `models/` directory:
- `opencv_face_detector.pbtxt`
- `opencv_face_detector_uint8.pb`
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

### 5. Configure Environment (Optional)

```bash
cp .env.example .env
# Edit .env with your settings
```

## Usage

### Run the API Server

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check

```bash
GET /health
```

### Process Video (Upload)

```bash
POST /process-video
Content-Type: multipart/form-data

# Upload video file
curl -X POST "http://localhost:8000/process-video" \
  -F "video=@path/to/video.mp4"
```

### Process Video (Path)

```bash
POST /process-video-path
Content-Type: application/json

{
  "video_path": "/path/to/video.mp4",
  "conf_threshold": 0.5
}
```

### Get All Detections

```bash
GET /detections
```

### Get Specific Detection

```bash
GET /detections/{detection_id}
```

### Get Face Image

```bash
GET /face-image/{person_id}
```

### Download Output Video

```bash
GET /download-output/{filename}
```

### Clear All Detections

```bash
DELETE /detections
```

## Response Examples

### Detection Response

```json
{
  "id": 1,
  "image": "facess/person_1.jpg",
  "gender": "Male",
  "age": "(20-29)",
  "entry_time": "2026-01-09 15:30:45"
}
```

### Video Process Response

```json
{
  "status": "success",
  "message": "Video processed successfully. Detected 5 unique persons.",
  "output_video": "outputs/output_video.mp4",
  "detections_count": 5,
  "detections": [...]
}
```

## Configuration

Edit `src/config.py` or use environment variables:

- `FACE_CONF_THRESHOLD`: Confidence threshold for face detection (default: 0.5)
- `PADDING`: Padding around detected faces (default: 20)
- `FRAMES_TO_STABILIZE`: Frames for age/gender stabilization (default: 3)
- `MAX_AGE`: Maximum age for tracker (default: 5)

## Development

### Project Structure

- **src/config.py**: Application configuration and settings
- **src/detector.py**: Face detection and age/gender prediction
- **src/processor.py**: Video processing and tracking logic
- **src/models.py**: Pydantic models for API requests/responses
- **src/main.py**: FastAPI application with endpoints

### Adding New Features

1. Add configuration in `src/config.py`
2. Implement logic in appropriate module
3. Add endpoint in `src/main.py`
4. Update models in `src/models.py`

## Dependencies

- **FastAPI**: Web framework
- **OpenCV**: Computer vision
- **DeepSORT**: Object tracking
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

## Troubleshooting

### Model Files Not Found

Make sure all model files are in the `models/` directory with correct names.

### Video Processing Fails

- Check video file format (MP4 recommended)
- Ensure sufficient disk space
- Verify model files are loaded correctly

### API Not Starting

- Check if port 8000 is available
- Verify all dependencies are installed
- Check model files exist

## Production Deployment

### Using Gunicorn + Uvicorn

```bash
gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t gender-age-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models gender-age-api
```

## License

MIT License

## Author

ِSir Ahmed 🤍 - 2026
