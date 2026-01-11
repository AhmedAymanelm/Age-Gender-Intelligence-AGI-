"""Test script to process a video file directly."""
import sys
from pathlib import Path
from src.processor import VideoProcessor
from src.config import settings

def test_video(video_path: str):
    """Test video processing."""
    video_file = Path(video_path)
    
    if not video_file.exists():
        print(f"❌ Error: Video file not found: {video_path}")
        return
    
    print(f"🎥 Testing video: {video_file.name}")
    print(f"📁 Video path: {video_path}")
    print("-" * 50)
    
    try:
        # Initialize processor
        print("🔄 Initializing video processor...")
        processor = VideoProcessor()
        
        # Process video
        output_name = f"output_{video_file.stem}.mp4"
        result = processor.process_video(str(video_path), output_name)
        
        print("-" * 50)
        print("✅ Processing completed!")
        print(f"📊 Results:")
        print(f"   - Output video: {settings.output_dir / output_name}")
        print(f"   - Data file: {settings.data_file}")
        print(f"   - Faces folder: {settings.faces_dir}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    test_video(video_path)
