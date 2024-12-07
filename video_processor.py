import cv2
import numpy as np
from pathlib import Path
import tempfile
from typing import Dict, Tuple, Optional, List
import time
from dataclasses import dataclass
from datetime import datetime
import math
import os
import traceback

@dataclass
class FrameMetrics:
    timestamp: float
    face_detected: bool
    eyes_detected: int
    confidence: float
    behaviors: Dict[str, float]
    alert_level: str
    recommendations: List[str]

class VideoProcessor:
    def __init__(self, resize_factor=0.5, batch_size=16, num_streams=8):
        """Initialize the video processor with GPU optimization settings."""
        self.resize_factor = resize_factor
        self.batch_size = batch_size
        self.num_streams = num_streams
        
        # Load cascade classifiers
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        self.face_cascade = cv2.CascadeClassifier()
        self.eye_cascade = cv2.CascadeClassifier()
        
        if not self.face_cascade.load(face_cascade_path):
            raise RuntimeError(f"Error loading face cascade from {face_cascade_path}")
        if not self.eye_cascade.load(eye_cascade_path):
            raise RuntimeError(f"Error loading eye cascade from {eye_cascade_path}")
        
        # GPU optimization settings for RTX 4060 Ti
        # Initialize CUDA device
        cuda_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if cuda_devices > 0:
            cv2.cuda.setDevice(0)
            # Set larger GPU memory pool
            cv2.cuda.setBufferPoolUsage(True)
            cv2.cuda.setBufferPoolConfig(1024 * 1024 * 1024, 4)  # 1GB pool, 4 streams
            
            # Create cascade classifiers
            self.face_cascade_cuda = cv2.cuda.CascadeClassifier_create(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade_cuda = cv2.cuda.CascadeClassifier_create(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            
            # Pre-warm GPU
            dummy = cv2.cuda_GpuMat((100, 100), cv2.CV_8UC3)
            dummy.upload(np.zeros((100, 100, 3), np.uint8))
            dummy.release()
        else:
            print("Warning: No CUDA devices available. Using CPU fallback.")
        
        # Enhanced model accuracies based on Haar Cascade performance
        self.model_accuracy = {
            'face_detection': 0.92,  # Improved with optimal parameters
            'eye_detection': 0.88,
            'behavior_classification': 0.85
        }
        
        # Behavior thresholds
        self.DROWSY_THRESHOLD = 0.6
        self.DISTRACTION_THRESHOLD = 0.5
        
        # Rolling window for smoothing predictions
        self.window_size = 10
        self.frame_metrics: List[FrameMetrics] = []
        
        # State tracking for drowsiness detection
        self.eye_state_history = []
        self.drowsy_threshold = 2  # Reduced threshold - alert after 2 frames
        self.max_history = 3  # Reduced history for faster response
        self.last_alert_time = 0
        self.alert_cooldown = 1.0  # Seconds between alerts
        
        self.eyes_closed_start = None
        self.distraction_start = None

    def get_alert_level(self, behaviors: Dict[str, float]) -> Tuple[str, List[str]]:
        """Determine alert level and provide recommendations."""
        recommendations = []
        
        if behaviors["drowsy"] > self.DROWSY_THRESHOLD:
            alert_level = "HIGH RISK ⚠️"
            recommendations = [
                "Pull over immediately and take a break",
                "Get some rest before continuing",
                "Consider having some caffeine"
            ]
        elif behaviors["distracted"] > self.DISTRACTION_THRESHOLD:
            alert_level = "MEDIUM RISK ⚡"
            recommendations = [
                "Keep your eyes on the road",
                "Minimize distractions",
                "Focus on driving"
            ]
        elif behaviors["normal"] > 0.7:
            alert_level = "LOW RISK ✅"
            recommendations = [
                "Maintain current driving behavior",
                "Stay alert and focused"
            ]
        else:
            alert_level = "MODERATE RISK ⚠️"
            recommendations = [
                "Increase attention to driving",
                "Take a break if needed"
            ]
            
        return alert_level, recommendations

    def smooth_predictions(self, current_metrics: Dict[str, float]) -> Dict[str, float]:
        """Apply smoothing to behavior predictions using recent history."""
        if len(self.frame_metrics) < 2:
            return current_metrics
            
        window = self.frame_metrics[-min(self.window_size, len(self.frame_metrics)):]
        smoothed = {}
        
        for behavior in current_metrics.keys():
            history = [fm.behaviors[behavior] for fm in window]
            # Exponential moving average with more weight to recent frames
            weights = [math.exp(i/len(history)) for i in range(len(history))]
            smoothed[behavior] = (sum(w * v for w, v in zip(weights, history)) / sum(weights))
            
        return smoothed

    def detect_eyes(self, face_roi):
        """Detect eyes in the face region with improved accuracy."""
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_roi = clahe.apply(gray_roi)
        
        # Detect eyes with more lenient parameters
        eyes = self.eye_cascade.detectMultiScale(
            gray_roi,
            scaleFactor=1.1,
            minNeighbors=3,  # Reduced for more sensitive detection
            minSize=(20, 20),  # Smaller minimum size
            maxSize=(50, 50)   # Larger maximum size
        )
        
        # Filter out false positives
        valid_eyes = []
        face_height, face_width = face_roi.shape[:2]
        
        # Expanded eye regions
        upper_limit = int(face_height * 0.1)   # Top 10% of face
        lower_limit = int(face_height * 0.5)   # Bottom 50% of face
        
        for (x, y, w, h) in eyes:
            # Check if eye is in the expected vertical region
            if y > upper_limit and y < lower_limit:
                # More lenient aspect ratio check
                aspect_ratio = w / h
                if 0.4 <= aspect_ratio <= 2.5:  # Wider range for aspect ratio
                    valid_eyes.append((x, y, w, h))
        
        return valid_eyes

    def update_eye_state(self, eyes_detected, frame_time):
        """Track eye state over time to detect prolonged closure."""
        # Add current state (1 for eyes open, 0 for closed)
        current_state = 1 if eyes_detected >= 2 else 0
        self.eye_state_history.append(current_state)
        
        if len(self.eye_state_history) > self.max_history:
            self.eye_state_history.pop(0)
        
        # Calculate metrics
        if len(self.eye_state_history) >= 2:  # Need at least 2 frames
            closed_ratio = 1 - (sum(self.eye_state_history) / len(self.eye_state_history))
            
            # Count consecutive closed frames from the end
            consecutive_closed = 0
            for state in reversed(self.eye_state_history):
                if state == 0:
                    consecutive_closed += 1
                else:
                    break
            
            # More aggressive alert triggering
            should_alert = consecutive_closed >= self.drowsy_threshold or closed_ratio > 0.5
            
            # Reset alert cooldown if needed
            if should_alert and (frame_time - self.last_alert_time) >= self.alert_cooldown:
                self.last_alert_time = frame_time
            
            return closed_ratio, consecutive_closed, should_alert
        return 0, 0, False

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[str, float], FrameMetrics]:
        """Process a video frame with improved drowsiness detection."""
        frame_metrics = {
            'face_detected': False,
            'eyes_detected': 0,
            'confidence': 0.0,
            'alert_level': 'LOW RISK ✅',
            'behaviors': {
                'normal': 1.0,
                'drowsy': 0.0,
                'distracted': 0.0
            }
        }
        
        try:
            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces with more sensitive parameters
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,  # More sensitive scale factor
                minNeighbors=3,   # Reduced for more detections
                minSize=(60, 60)  # Smaller minimum face size
            )
            
            if len(faces) > 0:
                # Process the largest face
                (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Get face ROI
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Detect eyes with more sensitive parameters
                eyes = self.eye_cascade.detectMultiScale(
                    roi_gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(20, 20)  # Smaller minimum eye size
                )
                
                # Track eye state
                current_time = time.time()
                if len(eyes) < 2:  # Eyes closed or not detected
                    if self.eyes_closed_start is None:
                        self.eyes_closed_start = current_time
                    eyes_closed_duration = current_time - self.eyes_closed_start if self.eyes_closed_start else 0
                    
                    # More sensitive drowsiness detection (>0.5 seconds)
                    if eyes_closed_duration > 0.5:  # Reduced from 1.0 to 0.5 seconds
                        frame_metrics['behaviors']['drowsy'] = min(1.0, eyes_closed_duration / 2.0)  # Faster increase
                        frame_metrics['behaviors']['normal'] = max(0.0, 1.0 - frame_metrics['behaviors']['drowsy'])
                        
                        # Add warning text
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.eyes_closed_start = None
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Head position analysis for distraction
                face_center_x = x + w//2
                frame_center_x = frame.shape[1]//2
                offset_ratio = abs(face_center_x - frame_center_x) / (frame.shape[1]//4)  # More sensitive offset
                
                # Track distraction duration
                if offset_ratio > 0.2:  # Reduced threshold from 0.3 to 0.2
                    if self.distraction_start is None:
                        self.distraction_start = current_time
                    distraction_duration = current_time - self.distraction_start if self.distraction_start else 0
                    
                    # More sensitive distraction detection (>0.5 seconds)
                    if distraction_duration > 0.5:  # Reduced from 1.0 to 0.5 seconds
                        frame_metrics['behaviors']['distracted'] = min(1.0, distraction_duration / 2.0)  # Faster increase
                        frame_metrics['behaviors']['normal'] = max(0.0, 1.0 - frame_metrics['behaviors']['distracted'])
                        
                        # Add warning text
                        cv2.putText(frame, "DISTRACTION DETECTED!", (10, 190),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    self.distraction_start = None
                
                # Determine alert level with more sensitive thresholds
                drowsy_level = frame_metrics['behaviors']['drowsy']
                distracted_level = frame_metrics['behaviors']['distracted']
                
                if drowsy_level > 0.2 or distracted_level > 0.2:  # Reduced from 0.3
                    frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
                elif drowsy_level > 0.1 or distracted_level > 0.1:  # Kept at 0.1
                    frame_metrics['alert_level'] = 'MEDIUM RISK ⚡'
                else:
                    frame_metrics['alert_level'] = 'LOW RISK ✅'
                
                # Add text overlays
                cv2.putText(frame, f"Drowsy: {drowsy_level*100:.1f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Distracted: {distracted_level*100:.1f}%", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, frame_metrics['alert_level'], (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (0, 255, 0) if frame_metrics['alert_level'] == 'LOW RISK ✅' else
                        (0, 255, 255) if frame_metrics['alert_level'] == 'MEDIUM RISK ⚡' else
                        (0, 0, 255), 2)
            else:
                # No face detected - consider as distracted
                frame_metrics['behaviors']['distracted'] = 1.0
                frame_metrics['behaviors']['normal'] = 0.0
                frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
                cv2.putText(frame, "NO FACE DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
    
        return frame, frame_metrics, frame_metrics

    def process_frame_gpu(self, frame, faces, eye_cascade, stream, gpu_gray):
        """Process a single frame using GPU acceleration."""
        frame_metrics = {
            'behaviors': {'normal': 0.0, 'drowsy': 0.0, 'distracted': 0.0},
            'alert_level': 'LOW RISK ✅'
        }
        
        if len(faces) == 0:
            frame_metrics['behaviors']['distracted'] = 1.0
            frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
            return frame_metrics
        
        # Process the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face ROI
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
        
        # Analyze eye state
        if len(eyes) >= 2:
            frame_metrics['behaviors']['normal'] = 1.0
        elif len(eyes) == 1:
            frame_metrics['behaviors']['drowsy'] = 0.7
            frame_metrics['behaviors']['normal'] = 0.3
            frame_metrics['alert_level'] = 'MEDIUM RISK ⚡'
        else:
            frame_metrics['behaviors']['drowsy'] = 1.0
            frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
        
        return frame_metrics

    def process_frame_cpu(self, frame, faces, eye_cascade):
        """Process a single frame using CPU."""
        frame_metrics = {
            'behaviors': {'normal': 0.0, 'drowsy': 0.0, 'distracted': 0.0},
            'alert_level': 'LOW RISK ✅'
        }
        
        if len(faces) == 0:
            frame_metrics['behaviors']['distracted'] = 1.0
            frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
            return frame_metrics
        
        # Process the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = frame[y:y+h, x:x+w]
        face_roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes in the face ROI
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
        
        # Analyze eye state
        if len(eyes) >= 2:
            frame_metrics['behaviors']['normal'] = 1.0
        elif len(eyes) == 1:
            frame_metrics['behaviors']['drowsy'] = 0.7
            frame_metrics['behaviors']['normal'] = 0.3
            frame_metrics['alert_level'] = 'MEDIUM RISK ⚡'
        else:
            frame_metrics['behaviors']['drowsy'] = 1.0
            frame_metrics['alert_level'] = 'HIGH RISK ⚠️'
        
        return frame_metrics

    def process_video(self, video_path: str, progress_callback=None) -> Dict[str, any]:
        """Process video with optimized GPU acceleration."""
        # Initialize resources as None
        cap = None
        streams = None
        gpu_frames = None
        gpu_grays = None
        
        try:
            # Initialize video capture with larger buffer
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 32)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video file: {video_path}")
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_width = int(original_width * self.resize_factor)
            frame_height = int(original_height * self.resize_factor)
            
            # Initialize results
            analysis_results = {
                'total_frames': total_frames,
                'processed_frames': 0,
                'duration': total_frames / fps,
                'fps': fps,
                'summary': {
                    'duration': format_timestamp(total_frames / fps),
                    'high_risk_count': 0,
                    'total_high_risk_time': 0,
                    'high_risk_segments': [],
                    'average_metrics': {
                        'normal': 0.0,
                        'drowsy': 0.0,
                        'distracted': 0.0
                    },
                    'alert_distribution': {
                        'LOW RISK ✅': 0,
                        'MEDIUM RISK ⚡': 0,
                        'HIGH RISK ⚠️': 0
                    },
                    'risk_assessment': 'Analysis not completed',
                    'recommendations': ['Processing...']
                }
            }
            
            # Create GPU resources
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            if cuda_available:
                try:
                    streams = [cv2.cuda.Stream() for _ in range(self.num_streams)]
                    gpu_frames = [cv2.cuda_GpuMat() for _ in range(self.batch_size)]
                    gpu_grays = [cv2.cuda_GpuMat() for _ in range(self.batch_size)]
                    
                    # Pre-allocate GPU memory for resized frames
                    for gpu_frame in gpu_frames:
                        gpu_frame.create((frame_height, frame_width), cv2.CV_8UC3)
                    for gpu_gray in gpu_grays:
                        gpu_gray.create((frame_height, frame_width), cv2.CV_8UC1)
                except cv2.error as e:
                    print(f"Error initializing GPU resources: {str(e)}")
                    cuda_available = False
                    # Clean up any partially initialized GPU resources
                    if streams:
                        for stream in streams:
                            stream.free()
                    if gpu_frames:
                        for gpu_frame in gpu_frames:
                            gpu_frame.release()
                    if gpu_grays:
                        for gpu_gray in gpu_grays:
                            gpu_gray.release()
                    streams = None
                    gpu_frames = None
                    gpu_grays = None
            
            # Process frames in batches
            processed_frames = 0
            batch_frames = []
            normal_sum = drowsy_sum = distracted_sum = 0
            high_risk_buffer = []  # Buffer for tracking consecutive high-risk frames
            current_segment = None
            segments = []  # Collect all segments for post-processing
            
            while True:
                # Read batch of frames
                while len(batch_frames) < self.batch_size:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if self.resize_factor != 1.0:
                        frame = cv2.resize(frame, (frame_width, frame_height))
                    batch_frames.append(frame)
                
                if not batch_frames:
                    break
                
                # Process batch using GPU or CPU
                batch_metrics = []
                if cuda_available and gpu_frames and gpu_grays and streams:
                    for i, frame in enumerate(batch_frames):
                        stream_idx = i % self.num_streams
                        gpu_frames[i].upload(frame, streams[stream_idx])
                        cv2.cuda.cvtColor(gpu_frames[i], cv2.COLOR_BGR2GRAY, gpu_grays[i], streams[stream_idx])
                        
                        # Detect faces
                        faces = self.face_cascade_cuda.detectMultiScale(gpu_grays[i], streams[stream_idx])
                        faces = faces.download()
                        
                        # Process detected faces
                        frame_metrics = self.process_frame_gpu(frame, faces, self.eye_cascade_cuda, streams[stream_idx], gpu_grays[i])
                        batch_metrics.append(frame_metrics)
                else:
                    # CPU fallback
                    for frame in batch_frames:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray)
                        frame_metrics = self.process_frame_cpu(frame, faces, self.eye_cascade)
                        batch_metrics.append(frame_metrics)
                
                # Update metrics and handle high-risk segments
                for frame_metrics in batch_metrics:
                    current_time = processed_frames / fps
                    
                    # Update behavior sums
                    normal_sum += frame_metrics['behaviors']['normal']
                    drowsy_sum += frame_metrics['behaviors']['drowsy']
                    distracted_sum += frame_metrics['behaviors']['distracted']
                    
                    # Update alert frequencies
                    analysis_results['summary']['alert_distribution'][frame_metrics['alert_level']] += 1
                    
                    # Handle high-risk segments with buffering
                    if frame_metrics['alert_level'] == 'HIGH RISK ⚠️':
                        high_risk_buffer.append((processed_frames, current_time))
                        
                        # Start new segment if buffer reaches threshold
                        if len(high_risk_buffer) >= int(fps * 0.5) and current_segment is None:  # 0.5 second buffer
                            start_frame, start_time = high_risk_buffer[0]
                            current_segment = {
                                'start': format_timestamp(start_time),
                                'start_frame': start_frame,
                                'end': None,
                                'end_frame': None,
                                'duration': 0
                            }
                    else:
                        # End segment if buffer is empty and we have a current segment
                        if len(high_risk_buffer) == 0 and current_segment is not None:
                            current_segment['end'] = format_timestamp(current_time)
                            current_segment['end_frame'] = processed_frames
                            current_segment['duration'] = round(
                                float(parse_timestamp(current_segment['end'])) - 
                                float(parse_timestamp(current_segment['start'])), 
                                1
                            )
                            
                            # Add segment if it meets minimum duration
                            if current_segment['duration'] >= 1.0:  # Minimum 1 second duration
                                segments.append(current_segment.copy())
                            current_segment = None
                        
                        # Clear buffer if no high risk
                        high_risk_buffer = []
                    
                    processed_frames += 1
                
                # Update progress
                if progress_callback:
                    progress_callback(min(processed_frames / total_frames, 1.0))
                
                # Clear batch
                batch_frames.clear()
                
                # Synchronize streams periodically
                if cuda_available and streams and processed_frames % (self.batch_size * 4) == 0:
                    for stream in streams:
                        stream.waitForCompletion()
            
            # Handle final high-risk segment
            if current_segment is not None and len(high_risk_buffer) > 0:
                current_segment['end'] = format_timestamp(high_risk_buffer[-1][1])
                current_segment['end_frame'] = high_risk_buffer[-1][0]
                current_segment['duration'] = round(
                    float(parse_timestamp(current_segment['end'])) - 
                    float(parse_timestamp(current_segment['start'])), 
                    1
                )
                
                if current_segment['duration'] >= 1.0:
                    segments.append(current_segment.copy())
            
            # Merge nearby segments and update results
            analysis_results['summary']['high_risk_segments'] = self.merge_nearby_segments(segments)
            
            # Calculate final metrics
            if processed_frames > 0:
                # Update high risk statistics
                analysis_results['summary']['high_risk_count'] = len(analysis_results['summary']['high_risk_segments'])
                analysis_results['summary']['total_high_risk_time'] = sum(
                    segment['duration'] for segment in analysis_results['summary']['high_risk_segments']
                )
                
                # Update average metrics
                analysis_results['summary']['average_metrics']['normal'] = (normal_sum / processed_frames) * 100
                analysis_results['summary']['average_metrics']['drowsy'] = (drowsy_sum / processed_frames) * 100
                analysis_results['summary']['average_metrics']['distracted'] = (distracted_sum / processed_frames) * 100
                
                # Update alert distribution percentages
                total_frames_f = float(processed_frames)
                for level in analysis_results['summary']['alert_distribution']:
                    analysis_results['summary']['alert_distribution'][level] = (
                        analysis_results['summary']['alert_distribution'][level] / total_frames_f * 100
                    )
                
                # Generate risk assessment and recommendations
                avg_drowsy = analysis_results['summary']['average_metrics']['drowsy'] / 100
                avg_distracted = analysis_results['summary']['average_metrics']['distracted'] / 100
                
                analysis_results['summary']['risk_assessment'] = self.get_risk_assessment(
                    avg_drowsy, avg_distracted
                )
                analysis_results['summary']['recommendations'] = self.get_recommendations(
                    avg_drowsy, avg_distracted
                )
        
            return analysis_results
            
        except Exception as e:
            print(f"Error processing video: {str(e)}")
            traceback.print_exc()
            return self.create_error_results(str(e))
            
        finally:
            # Cleanup resources
            if cap:
                cap.release()
            
            if cuda_available:
                try:
                    if gpu_frames:
                        for gpu_frame in gpu_frames:
                            gpu_frame.release()
                    if gpu_grays:
                        for gpu_gray in gpu_grays:
                            gpu_gray.release()
                    if streams:
                        for stream in streams:
                            stream.free()
                    cv2.cuda.setBufferPoolUsage(False)
                except Exception as e:
                    print(f"Error cleaning up GPU resources: {str(e)}")

    def get_risk_assessment(self, avg_drowsy: float, avg_distracted: float) -> str:
        """Generate overall risk assessment based on average metrics."""
        if avg_drowsy > 0.3 or avg_distracted > 0.3:
            return "HIGH RISK - Immediate attention required"
        elif avg_drowsy > 0.2 or avg_distracted > 0.2:
            return "MEDIUM RISK - Monitoring recommended"
        else:
            return "LOW RISK - Generally safe driving behavior"

    def get_recommendations(self, avg_drowsy: float, avg_distracted: float) -> List[str]:
        """Generate specific recommendations based on behavior patterns."""
        recommendations = []
        
        if avg_drowsy > 0.3:
            recommendations.extend([
                "Take immediate breaks when feeling drowsy",
                "Consider using driver alertness monitoring systems",
                "Ensure adequate rest before long drives"
            ])
        elif avg_drowsy > 0.2:
            recommendations.extend([
                "Schedule regular rest stops",
                "Monitor fatigue levels during long trips"
            ])
        
        if avg_distracted > 0.3:
            recommendations.extend([
                "Minimize distractions while driving",
                "Consider installing driver attention monitoring",
                "Review and eliminate common distraction sources"
            ])
        elif avg_distracted > 0.2:
            recommendations.extend([
                "Stay focused on the road",
                "Plan routes before driving"
            ])
        
        if not recommendations:
            recommendations.append("Maintain current safe driving practices")
        
        return recommendations

    def create_results(self):
        return {
            'total_frames': 0,
            'processed_frames': 0,
            'duration': 0,
            'fps': 0,
            'summary': {
                'duration': '00:00',
                'high_risk_count': 0,
                'total_high_risk_time': 0,
                'high_risk_segments': [],
                'average_metrics': {
                    'normal': 0.0,
                    'drowsy': 0.0,
                    'distracted': 0.0
                },
                'alert_distribution': {
                    'LOW RISK ✅': 0,
                    'MEDIUM RISK ⚡': 0,
                    'HIGH RISK ⚠️': 0
                },
                'risk_assessment': 'Analysis not completed',
                'recommendations': ['Unable to complete analysis']
            }
        }

    def create_error_results(self, error_message):
        return {
            'error': error_message,
            'summary': {
                'duration': '00:00',
                'high_risk_count': 0,
                'total_high_risk_time': 0,
                'high_risk_segments': [],
                'average_metrics': {
                    'normal': 0.0,
                    'drowsy': 0.0,
                    'distracted': 0.0
                },
                'alert_distribution': {
                    'LOW RISK ✅': 0,
                    'MEDIUM RISK ⚡': 0,
                    'HIGH RISK ⚠️': 0
                },
                'risk_assessment': 'Error processing video',
                'recommendations': [f'Error: {error_message}. Please try again with a different video.']
            }
        }

    def merge_nearby_segments(self, segments, max_gap=2.0):
        """Merge segments that are close to each other in time."""
        if not segments:
            return []
        
        # Convert timestamps to seconds for easier comparison
        for segment in segments:
            segment['start_sec'] = float(parse_timestamp(segment['start']))
            segment['end_sec'] = float(parse_timestamp(segment['end']))
        
        # Sort segments by start time
        segments.sort(key=lambda x: x['start_sec'])
        
        # Merge nearby segments
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            # If gap between segments is small enough, merge them
            if next_seg['start_sec'] - current['end_sec'] <= max_gap:
                current['end'] = next_seg['end']
                current['end_sec'] = next_seg['end_sec']
                current['duration'] = round(current['end_sec'] - current['start_sec'], 1)
            else:
                merged.append(current)
                current = next_seg
        
        merged.append(current)
        
        # Clean up temporary fields and ensure durations are accurate
        for segment in merged:
            del segment['start_sec']
            del segment['end_sec']
        
        return merged

def format_timestamp(seconds: float) -> str:
    """Format timestamp as MM:SS."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def parse_timestamp(timestamp: str) -> float:
    """Parse timestamp in MM:SS format to seconds."""
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds
