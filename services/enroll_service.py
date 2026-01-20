"""
Enrollment Service for Face Recognition Attendance System.
Handles user registration with face capture and embedding extraction.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from modules.face_detector import FaceDetector, DetectionStatus, get_face_detector
from modules.anti_spoofing import FaceAntiSpoofing, get_anti_spoofing
from modules.face_recognition import FaceRecognition, get_face_recognition
from config import config


@dataclass
class EnrollResult:
    """Result of enrollment process."""
    success: bool
    user_id: Optional[str]
    message: str
    embeddings_count: int
    frames_processed: int
    valid_frames: int


class EnrollmentService:
    """
    Enrollment Service for registering new users.
    
    Workflow:
    1. Collect frames from video stream
    2. For each frame: detect face, check quality, verify anti-spoofing
    3. Extract embeddings from valid frames
    4. Select diverse embeddings
    5. Store in database
    """
    
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        anti_spoofing: Optional[FaceAntiSpoofing] = None,
        face_recognition: Optional[FaceRecognition] = None,
        min_embeddings: int = 5,
        max_embeddings: int = 10,
        fas_threshold: float = 0.5
    ):
        """
        Initialize enrollment service.
        
        Args:
            face_detector: Face detector instance
            anti_spoofing: Anti-spoofing instance
            face_recognition: Face recognition instance
            min_embeddings: Minimum embeddings required for enrollment
            max_embeddings: Maximum embeddings to store
            fas_threshold: Anti-spoofing threshold
        """
        self.face_detector = face_detector or get_face_detector(
            min_face_size=config.MIN_FACE_SIZE,
            blur_threshold=config.MAX_BLUR_THRESHOLD,
            min_brightness=config.MIN_BRIGHTNESS,
            max_brightness=config.MAX_BRIGHTNESS
        )
        self.anti_spoofing = anti_spoofing or get_anti_spoofing(threshold=config.T_FAS)
        self.face_recognition = face_recognition or get_face_recognition(threshold=config.T_FR)
        
        self.min_embeddings = min_embeddings
        self.max_embeddings = max_embeddings
        self.fas_threshold = fas_threshold
    
    def process_frames(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[List[np.ndarray], List[np.ndarray], int]:
        """
        Process frames for enrollment.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            Tuple of (valid_face_crops, embeddings, valid_count)
        """
        valid_face_crops = []
        embeddings = []
        valid_count = 0
        
        for frame in frames:
            # Face detection
            det_result = self.face_detector.detect(frame)
            
            if det_result.status != DetectionStatus.OK:
                continue
            
            # Anti-spoofing check
            fas_result = self.anti_spoofing.check(det_result.face_crop)
            
            if not fas_result.is_real:
                logging.debug(f"Frame rejected: anti-spoofing score {fas_result.score:.3f}")
                continue
            
            # Extract embedding
            emb_result = self.face_recognition.extract_embedding(frame)
            
            if not emb_result.success:
                logging.debug(f"Embedding extraction failed: {emb_result.error}")
                continue
            
            valid_face_crops.append(det_result.face_crop)
            embeddings.append(emb_result.embedding)
            valid_count += 1
        
        return valid_face_crops, embeddings, valid_count
    
    def select_diverse_embeddings(
        self,
        embeddings: List[np.ndarray],
        count: int
    ) -> List[np.ndarray]:
        """
        Select diverse embeddings using farthest point sampling.
        
        Args:
            embeddings: List of embeddings
            count: Number to select
            
        Returns:
            Selected embeddings
        """
        if len(embeddings) <= count:
            return embeddings
        
        # Convert to numpy array
        emb_array = np.array(embeddings)
        n = len(emb_array)
        
        # Farthest point sampling
        selected_indices = [0]  # Start with first embedding
        
        while len(selected_indices) < count:
            # Calculate minimum distance to selected set for each point
            min_distances = np.full(n, np.inf)
            
            for idx in selected_indices:
                distances = np.linalg.norm(emb_array - emb_array[idx], axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # Mask already selected
            for idx in selected_indices:
                min_distances[idx] = -1
            
            # Select farthest point
            next_idx = np.argmax(min_distances)
            selected_indices.append(next_idx)
        
        return [embeddings[i] for i in selected_indices]
    
    async def enroll_from_frames(
        self,
        frames: List[np.ndarray],
        user_id: str,
        name: str,
        code: str,
        session  # AsyncSession
    ) -> EnrollResult:
        """
        Enroll user from collected frames.
        
        Args:
            frames: List of BGR frames
            user_id: User ID
            name: User name
            code: User code
            session: Database session
            
        Returns:
            EnrollResult
        """
        from database import crud
        
        # Process frames
        face_crops, embeddings, valid_count = self.process_frames(frames)
        
        if len(embeddings) < self.min_embeddings:
            return EnrollResult(
                success=False,
                user_id=None,
                message=f"Not enough valid frames. Got {len(embeddings)}, need {self.min_embeddings}",
                embeddings_count=len(embeddings),
                frames_processed=len(frames),
                valid_frames=valid_count
            )
        
        # Select diverse embeddings
        selected_embeddings = self.select_diverse_embeddings(embeddings, self.max_embeddings)
        
        try:
            # Check if user code already exists
            existing_user = await crud.get_user_by_code(session, code)
            if existing_user:
                return EnrollResult(
                    success=False,
                    user_id=None,
                    message=f"User code '{code}' already exists",
                    embeddings_count=0,
                    frames_processed=len(frames),
                    valid_frames=valid_count
                )
            
            # Create user
            user = await crud.create_user(session, name=name, code=code)
            
            # Create face templates
            await crud.create_face_templates_batch(session, user.user_id, selected_embeddings)
            
            # Register in recognition module
            self.face_recognition.register_user(
                user_id=user.user_id,
                name=name,
                code=code,
                embeddings=selected_embeddings
            )
            
            return EnrollResult(
                success=True,
                user_id=user.user_id,
                message=f"Successfully enrolled user '{name}' with {len(selected_embeddings)} embeddings",
                embeddings_count=len(selected_embeddings),
                frames_processed=len(frames),
                valid_frames=valid_count
            )
            
        except Exception as e:
            logging.error(f"Enrollment failed: {e}")
            return EnrollResult(
                success=False,
                user_id=None,
                message=f"Database error: {str(e)}",
                embeddings_count=0,
                frames_processed=len(frames),
                valid_frames=valid_count
            )
    
    def enroll_from_video(
        self,
        video_path: str,
        user_id: str,
        name: str,
        code: str,
        max_frames: int = 90,  # ~3 seconds at 30fps
        skip_frames: int = 3   # Process every 3rd frame
    ) -> Tuple[List[np.ndarray], int]:
        """
        Extract frames from video for enrollment.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to extract
            skip_frames: Skip every N frames
            
        Returns:
            Tuple of (frames, total_frame_count)
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % skip_frames == 0:
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames, frame_count


# Singleton instance
_enroll_service: Optional[EnrollmentService] = None


def get_enrollment_service() -> EnrollmentService:
    """Get or create enrollment service singleton."""
    global _enroll_service
    if _enroll_service is None:
        _enroll_service = EnrollmentService(
            min_embeddings=config.MIN_ENROLL_EMBEDDINGS,
            max_embeddings=config.MAX_ENROLL_EMBEDDINGS,
            fas_threshold=config.T_FAS
        )
    return _enroll_service
