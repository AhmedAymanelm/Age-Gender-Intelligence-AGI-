"""Video processing utilities."""
import cv2
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import deque, Counter
from typing import List, Dict, Any, Optional
from deep_sort_realtime.deepsort_tracker import DeepSort

from .config import settings
from .detector import get_detector


class VideoProcessor:
    """Process video for face detection and tracking."""
    
    def __init__(self):
        """Initialize video processor."""
        self.detector = get_detector()
        self.tracker = DeepSort(max_age=settings.max_age)
        self.track_info: Dict[int, Dict[str, Any]] = {}
        self.captured_ids = set()
        self.current_id = 1
        self.detections_data = []
        
        # Load existing detections if available
        if settings.data_file.exists():
            with open(settings.data_file, "r") as f:
                self.detections_data = json.load(f)
                self.captured_ids = set(entry["id"] for entry in self.detections_data)
                self.current_id = max(self.captured_ids) + 1 if self.captured_ids else 1
    
    def process_video(self, video_path: str, output_name: str = "output_video.mp4") -> Dict[str, Any]:
        """
        Process video file for face detection and tracking.
        
        Args:
            video_path: Path to input video
            output_name: Name for output video
            
        Returns:
            Dictionary with processing results
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Setup video writer
        output_path = settings.output_dir / output_name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Finished processing video")
                break
            
            frame_count += 1
            result_img = self._process_frame(frame)
            out.write(result_img)
            
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        # Save detections
        with open(settings.data_file, "w") as f:
            json.dump(self.detections_data, f, indent=4)
        
        return {
            "output_video": str(output_path),
            "frames_processed": frame_count,
            "detections_count": len(self.detections_data),
            "detections": self.detections_data
        }
    
    def _process_frame(self, frame):
        """Process a single frame."""
        result_img, face_boxes = self.detector.highlight_face(frame)
        
        # Prepare detections for tracker
        detections_for_tracker = [
            [(x1, y1, x2-x1, y2-y1), 1.0, "face"]
            for x1, y1, x2, y2 in face_boxes
        ]
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Apply padding
            box_width = x2 - x1
            box_height = y2 - y1
            x1 = max(0, x1 - settings.padding)
            y1 = max(0, y1 - settings.padding)
            x2 = min(frame.shape[1]-1, x1 + box_width + 2*settings.padding)
            y2 = min(frame.shape[0]-1, y1 + box_height + 2*settings.padding)
            
            if track_id not in self.track_info:
                # New person detected
                self._handle_new_track(track_id, frame, x1, y1, x2, y2)
            else:
                # Update existing track
                self._update_track(track_id, frame, x1, y1, x2, y2)
            
            # Draw bounding box and label
            label = (f'ID:{self.track_info[track_id]["person_id"]} '
                    f'{self.track_info[track_id]["gender"]} '
                    f'{self.track_info[track_id]["age"]}')
            cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        
        return result_img
    
    def _handle_new_track(self, track_id: int, frame, x1: int, y1: int, x2: int, y2: int):
        """Handle new tracked person."""
        face = frame[y1:y2, x1:x2]
        gender, age = self.detector.predict_age_gender(face)
        
        # Check if this person was already captured by comparing existing saved faces
        person_id = self._find_matching_person(face)
        
        if person_id is None:
            # New unique person - save face image
            person_id = self.current_id
            entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = settings.faces_dir / f"person_{person_id}.jpg"
            cv2.imwrite(str(filename), face)
            
            # Save detection data
            self.detections_data.append({
                "id": person_id,
                "image": str(filename),
                "gender": gender,
                "age": age,
                "entry_time": entry_time
            })
            
            print(f"[✔] Captured Person {person_id}: Gender={gender}, Age={age}, Entry={entry_time}")
            self.current_id += 1
        
        # Initialize deques for stabilization
        gender_deque = deque([gender], maxlen=settings.frames_to_stabilize)
        age_deque = deque([age], maxlen=settings.frames_to_stabilize)
        
        # Update tracking info
        self.captured_ids.add(track_id)
        self.track_info[track_id] = {
            "box": [x1, y1, x2, y2],
            "gender_deque": gender_deque,
            "age_deque": age_deque,
            "gender": gender,
            "age": age,
            "person_id": person_id
        }
    
    def _update_track(self, track_id: int, frame, x1: int, y1: int, x2: int, y2: int):
        """Update existing track with new predictions."""
        self.track_info[track_id]["box"] = [x1, y1, x2, y2]
        face = frame[y1:y2, x1:x2]
        gender, age = self.detector.predict_age_gender(face)
        
        # Update deques
        self.track_info[track_id]["gender_deque"].append(gender)
        self.track_info[track_id]["age_deque"].append(age)
        
        # Get most common values for stabilization
        self.track_info[track_id]["gender"] = Counter(
            self.track_info[track_id]["gender_deque"]
        ).most_common(1)[0][0]
        self.track_info[track_id]["age"] = Counter(
            self.track_info[track_id]["age_deque"]
        ).most_common(1)[0][0]
    
    def _find_matching_person(self, face: np.ndarray) -> int:
        """
        Find if this face matches any previously saved person.
        Returns person_id if match found, None otherwise.
        Uses simple face comparison based on histogram similarity.
        """
        if len(self.detections_data) == 0:
            return None
        
        # Resize face for comparison
        try:
            face_resized = cv2.resize(face, (100, 100))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        except:
            return None
        
        # Compare with all saved faces
        for detection in self.detections_data:
            saved_face_path = detection["image"]
            if not Path(saved_face_path).exists():
                continue
            
            saved_face = cv2.imread(str(saved_face_path))
            if saved_face is None:
                continue
            
            try:
                saved_face_resized = cv2.resize(saved_face, (100, 100))
                saved_face_gray = cv2.cvtColor(saved_face_resized, cv2.COLOR_BGR2GRAY)
                
                # Calculate histogram similarity
                hist1 = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([saved_face_gray], [0], None, [256], [0, 256])
                similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                # If similarity is high enough, consider it the same person
                if similarity > 0.85:  # Threshold can be adjusted
                    return detection["id"]
            except:
                continue
        
        return None
    
    def reset(self):
        """Reset processor state."""
        self.tracker = DeepSort(max_age=settings.max_age)
        self.track_info = {}
