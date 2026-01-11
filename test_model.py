"""Simple test script to verify model loading."""
import cv2
from src.config import settings

print("Testing model loading...")
print(f"Face proto: {settings.face_proto}")
print(f"Face model: {settings.face_model}")
print(f"Proto exists: {settings.face_proto.exists()}")
print(f"Model exists: {settings.face_model.exists()}")

try:
    face_net = cv2.dnn.readNetFromCaffe(
        str(settings.face_proto),
        str(settings.face_model)
    )
    print("✅ Face net loaded successfully")
    print(f"Empty: {face_net.empty()}")
    
    # Try a test inference
    import numpy as np
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(test_frame, 1.0, (300, 300), [104, 117, 123], True, False)
    print(f"Blob shape: {blob.shape}")
    
    face_net.setInput(blob)
    print("✅ setInput successful")
    
    detections = face_net.forward()
    print(f"✅ forward successful, detections shape: {detections.shape}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
