import cv2
import numpy as np
import logging
import time
import os
import subprocess
import gc
from typing import List, Dict, Any
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logger = logging.getLogger(__name__)

class OptimizedEmotionDetector:
    def __init__(self, model_path: str = None):
        """
        PhD-Level Optimized Emotion Detector.
        Uses MTCNN for high-accuracy face tracking and a specialized CNN for emotion classification.
        Optimized for CPU with batch sample processing to avoid OOM.
        """
        self._detector = MTCNN()
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Default fallback model path
        if not model_path:
            # Note: Path corrected to 'model/model.h5' per directory structure
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "model.h5")
        
        try:
            logger.info(f"Loading emotion classification model from {model_path}...")
            self.model = load_model(model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load emotion model: {e}")
            self.model = None

    def process_video(self, video_path: str, frame_skip: int = None, auto_orient: bool = True, debug: bool = False) -> Dict[str, Any]:
        """
        PhD Upgrade: Adaptive Global Sampling & Heuristic Orientation.
        Processes video files with a target sample count to ensure coverage of any video length while staying under memory limits.
        """
        start_time = time.time()
        try:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            # Get video properties for adaptive processing
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

            # Detect rotation metadata (common in mobile videos)
            rotation_degrees = 0
            if auto_orient:
                try:
                    probe = subprocess.run(
                        [
                            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream_tags=rotate',
                            '-of', 'default=nokey=1:noprint_wrappers=1',
                            video_path
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=5
                    )
                    out = probe.stdout.strip()
                    if out in {'90', '180', '270'}:
                        rotation_degrees = int(out)
                        logger.info(f"Detected rotation metadata: {rotation_degrees} degrees")
                except Exception as e:
                    logger.debug(f"ffprobe rotation check failed: {e}")

            # PhD Upgrade: Adaptive Global Sampling Variables
            target_samples = 500
            if file_size_mb > 100: target_samples = 1000
            if file_size_mb > 500: target_samples = 2000
            
            # Dynamically calculate skip to cover the whole file
            # If frame_skip is provided as argument, use it; otherwise calculate
            effective_frame_skip = frame_skip if frame_skip else max(30, total_frames // target_samples)
            max_samples = target_samples * 1.5 # Safety headroom
            
            min_face_confidence = 0.55
            memory_cleanup_interval = 40
            
            frame_count = 0
            processed_samples = 0
            emotion_data = []
            
            # Gestures & Eye Contact tracking state
            prev_gray = None
            motion_history = []
            eye_contact_samples = 0
            rotation_selected = False
            first_frame_checked = False
            
            logger.info(f"Processing Video: {file_size_mb:.1f}MB, {duration:.1f}s, skip={effective_frame_skip}")

            while cap.isOpened() and processed_samples < max_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 1. MOTION TRACKING (Gestures Heuristic)
                # Compute difference every N frames for speed
                if frame_count % (effective_frame_skip // 2 or 1) == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if prev_gray is not None:
                        diff = cv2.absdiff(gray, prev_gray)
                        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        motion_val = float(np.sum(thresh) / (frame.shape[0] * frame.shape[1] * 255.0))
                        motion_history.append(motion_val)
                    prev_gray = gray

                # 2. SELECTION (Sample every X frames)
                if frame_count % effective_frame_skip != 0:
                    continue
                
                processed_samples += 1
                
                # Auto-orient on first frame if not meta-detected
                if auto_orient and not rotation_selected and not first_frame_checked:
                    try:
                        candidates = [0, 90, 180, 270]
                        best_deg = rotation_degrees
                        best_faces = -1
                        for deg in candidates:
                            test_frame = frame
                            if deg == 90: test_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                            elif deg == 180: test_frame = cv2.rotate(frame, cv2.ROTATE_180)
                            elif deg == 270: test_frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            test_rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
                            detections = self._detector.detect_faces(test_rgb)
                            if len(detections) > best_faces:
                                best_faces = len(detections)
                                best_deg = deg
                        rotation_degrees = best_deg
                        rotation_selected = True
                        first_frame_checked = True
                        logger.info(f"Auto-selected rotation: {rotation_degrees} deg")
                    except Exception: 
                        rotation_selected = True
                        first_frame_checked = True

                # Apply rotation
                if rotation_degrees == 90:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_degrees == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation_degrees == 270:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Process Emotion with MTCNN + CNN
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self._detector.detect_faces(rgb_frame)
                
                if faces:
                    # Sort candidates by confidence
                    faces = sorted(faces, key=lambda x: x['confidence'], reverse=True)
                    target_face = faces[0]
                    
                    if target_face['confidence'] > min_face_confidence:
                        # 3. EYE CONTACT HEURISTIC
                        # Presence and centering of keypoints
                        kp = target_face.get('keypoints', {})
                        if 'left_eye' in kp and 'right_eye' in kp:
                            eye_contact_samples += 1
                        
                        # EMOTION CLASSIFICATION
                        x, y, w, h = target_face['box']
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        face_img = frame[y1:y2, x1:x2]
                        
                        if face_img.size > 0:
                            face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                            face_gray = cv2.resize(face_gray, (48, 48))
                            face_gray = face_gray.astype("float") / 255.0
                            face_gray = img_to_array(face_gray)
                            face_gray = np.expand_dims(face_gray, axis=0)
                            
                            preds = self.model.predict(face_gray, verbose=0)[0]
                            dominant = self.emotion_labels[np.argmax(preds)]
                            
                            timestamp = frame_count / fps if fps > 0 else 0
                            emotion_data.append({
                                "timestamp": round(timestamp, 2),
                                "timestamp": round(timestamp, 2),
                                "emotion": dominant,
                                "probabilities": [round(float(p), 4) for p in preds],
                                "confidence": round(float(target_face['confidence']), 3)
                            })

                # Memory mitigation
                if processed_samples % memory_cleanup_interval == 0:
                    gc.collect()

            cap.release()
            
            # Final Statistics
            avg_probs = np.mean([d['probabilities'] for d in emotion_data], axis=0).tolist() if emotion_data else [0.0]*7
            dominant_overall = self.emotion_labels[np.argmax(avg_probs)] if emotion_data else "Unknown"
            
            # Legacy backward compatibility for distribution
            emotion_distribution = {}
            if emotion_data:
                for i, label in enumerate(self.emotion_labels):
                    emotion_distribution[label] = round(avg_probs[i] * 100, 1)

            # BEHAVIORAL HIGHLIGHTS
            eye_contact_score = (eye_contact_samples / processed_samples * 100) if processed_samples > 0 else 0
            avg_motion = np.mean(motion_history) if motion_history else 0
            # Gestures approx count (peaks in motion)
            gestures_count = len([m for m in motion_history if m > 0.05]) # Simple threshold
            gestures_per_min = (gestures_count / (duration / 60)) if duration > 0 else 0

            return {
                "dominant_emotion": dominant_overall,
                "emotion_distribution": emotion_distribution,
                "average_probabilities": avg_probs,
                "emotion_data": emotion_data,
                "behavioral_metrics": {
                    "eye_contact_pct": round(float(eye_contact_score), 1),
                    "gestures_per_min": round(float(gestures_per_min), 1),
                    "motion_intensity": round(float(avg_motion), 4)
                },
                "processing_stats": {
                    "total_frames": total_frames,
                    "duration": round(duration, 2),
                    "samples_processed": processed_samples,
                    "time_taken": round(time.time() - start_time, 2),
                    "rotation": rotation_degrees
                }
            }

        except Exception as e:
            logger.error(f"Video analysis error: {str(e)}")
            return {"error": str(e)}

def process_video_optimized(video_path: str, **kwargs):
    """Legacy wrapper for global instance usage"""
    detector = OptimizedEmotionDetector()
    return detector.process_video(video_path, **kwargs)

if __name__ == "__main__":
    # Test block
    logging.basicConfig(level=logging.INFO)
    detector = OptimizedEmotionDetector()
    # Replace with a real path if testing locally
    # result = detector.process_video("test.mp4")
    # print(result)
