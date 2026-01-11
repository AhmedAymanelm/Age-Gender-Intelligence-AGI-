"""Face detection and age/gender prediction module."""
import cv2
import numpy as np
from typing import List, Tuple
from .config import settings


class FaceDetector:
    """Face detector using OpenCV DNN."""
    
    def __init__(self):
        """Initialize face detector."""
        self.conf_threshold = settings.face_conf_threshold
        self._load_models()
    
    def _load_models(self):
        """Load all required models."""
        try:
            # Check if all model files exist
            required_files = [
                settings.face_proto,
                settings.face_model,
                settings.age_proto,
                settings.age_model,
                settings.gender_proto,
                settings.gender_model
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    raise FileNotFoundError(f"Missing model file: {file_path}")
            
            # Load networks using explicit Caffe format
            self.face_net = cv2.dnn.readNetFromCaffe(
                str(settings.face_proto),
                str(settings.face_model)
            )
            self.age_net = cv2.dnn.readNetFromCaffe(
                str(settings.age_proto),
                str(settings.age_model)
            )
            self.gender_net = cv2.dnn.readNetFromCaffe(
                str(settings.gender_proto),
                str(settings.gender_model)
            )
            
            print("✓ All models loaded successfully")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    def highlight_face(self, frame: np.ndarray) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Detect faces in frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (frame, list of face boxes)
        """
        frame_opencv_dnn = frame.copy()
        frame_height = frame_opencv_dnn.shape[0]
        frame_width = frame_opencv_dnn.shape[1]
        
        blob = cv2.dnn.blobFromImage(
            frame_opencv_dnn, 1.0, (300, 300),
            [104, 117, 123], True, False
        )
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        face_boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                face_boxes.append([x1, y1, x2, y2])
        
        return frame_opencv_dnn, face_boxes
    
    def predict_age_gender(self, face: np.ndarray) -> Tuple[str, str]:
        """
        Predict age and gender from face crop.
        
        Args:
            face: Cropped face image
            
        Returns:
            Tuple of (gender, age)
        """
        if face.size == 0:
            return "Unknown", "Unknown"
        
        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            settings.model_mean_values,
            swapRB=False
        )
        
        # Predict gender
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = settings.gender_list[gender_preds[0].argmax()]
        
        # Predict age
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = settings.age_list[age_preds[0].argmax()]
        
        return gender, age


# Singleton instance
_detector_instance = None


def get_detector() -> FaceDetector:
    """Get or create detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance
