"""Download required model files for gender and age detection."""
import os
import urllib.request
from pathlib import Path

# Model URLs
MODELS = {
    "opencv_face_detector.pbtxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt",
    "opencv_face_detector_uint8.pb": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "age_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/age_net_definitions/age_deploy.prototxt",
    "age_net.caffemodel": "https://www.dropbox.com/s/xfb20y596869vbb/age_net.caffemodel?dl=1",
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/gender_net_definitions/gender_deploy.prototxt",
    "gender_net.caffemodel": "https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=1"
}

def download_file(url: str, destination: Path):
    """Download a file from URL to destination."""
    print(f"📥 Downloading {destination.name}...")
    try:
        urllib.request.urlretrieve(url, destination)
        print(f"✅ Downloaded {destination.name}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {destination.name}: {str(e)}")
        return False

def main():
    """Download all model files."""
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting model download...")
    print("=" * 60)
    
    success_count = 0
    total_count = len(MODELS)
    
    for filename, url in MODELS.items():
        destination = models_dir / filename
        
        # Skip if file already exists
        if destination.exists():
            print(f"⏭️  {filename} already exists, skipping...")
            success_count += 1
            continue
        
        if download_file(url, destination):
            success_count += 1
    
    print("=" * 60)
    print(f"✅ Download complete: {success_count}/{total_count} files")
    
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("You can now run: python test_video.py <video_path>")
    else:
        print("⚠️  Some models failed to download. Please download them manually.")
        print("See README.md for model sources.")

if __name__ == "__main__":
    main()
