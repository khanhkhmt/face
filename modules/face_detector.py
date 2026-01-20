"""
Face Detection Module using InsightFace/RetinaFace.
Handles face detection, quality checking, and face cropping.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None
    logging.warning("InsightFace not installed. Face detection will not work.")


class DetectionStatus(str, Enum):
    """Status codes for face detection."""
    OK = "OK"
    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"
    LOW_QUALITY = "LOW_QUALITY"


@dataclass
class QualityMetrics:
    """Face quality metrics."""
    face_size: int  # Size of face in pixels (width)
    blur_score: float  # Laplacian variance (higher = sharper)
    brightness: float  # Average brightness
    is_valid: bool
    rejection_reason: Optional[str] = None


@dataclass
class DetectionResult:
    """Result of face detection."""
    status: DetectionStatus
    face_crop: Optional[np.ndarray] = None
    bounding_box: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None
    quality_metrics: Optional[QualityMetrics] = None
    aligned_face: Optional[np.ndarray] = None  # 112x112 aligned face for recognition


class FaceDetector:
    """
    Face Detection module using InsightFace.
    
    Performs:
    1. Face detection
    2. Quality checking (size, blur, brightness)
    3. Face alignment for recognition
    """
    
    def __init__(
        self,
        min_face_size: int = 80,
        blur_threshold: float = 100.0,
        min_brightness: int = 40,
        max_brightness: int = 220,
        det_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize face detector.
        
        Args:
            min_face_size: Minimum face size in pixels
            blur_threshold: Minimum Laplacian variance (lower = more blur)
            min_brightness: Minimum average brightness
            max_brightness: Maximum average brightness
            det_size: Detection input size
        """
        self.min_face_size = min_face_size
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.det_size = det_size
        
        self._detector = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of the detector."""
        if self._initialized:
            return
        
        if FaceAnalysis is None:
            raise RuntimeError("InsightFace is not installed. Please install with: pip install insightface onnxruntime")
        
        self._detector = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self._detector.prepare(ctx_id=0, det_size=self.det_size)
        self._initialized = True
        logging.info("Face detector initialized successfully")
    
    def _calculate_blur(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate average brightness."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return np.mean(gray)
    
    def _check_quality(self, face_crop: np.ndarray, bbox: np.ndarray) -> QualityMetrics:
        """
        Check face quality.
        
        Args:
            face_crop: Cropped face image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            QualityMetrics with validity status
        """
        # Calculate face size (width)
        face_size = int(bbox[2] - bbox[0])
        
        # Calculate blur score
        blur_score = self._calculate_blur(face_crop)
        
        # Calculate brightness
        brightness = self._calculate_brightness(face_crop)
        
        # Check validity
        is_valid = True
        rejection_reason = None
        
        if face_size < self.min_face_size:
            is_valid = False
            rejection_reason = f"Face too small: {face_size}px < {self.min_face_size}px"
        elif blur_score < self.blur_threshold:
            is_valid = False
            rejection_reason = f"Face too blurry: {blur_score:.1f} < {self.blur_threshold}"
        elif brightness < self.min_brightness:
            is_valid = False
            rejection_reason = f"Face too dark: {brightness:.1f} < {self.min_brightness}"
        elif brightness > self.max_brightness:
            is_valid = False
            rejection_reason = f"Face too bright: {brightness:.1f} > {self.max_brightness}"
        
        return QualityMetrics(
            face_size=face_size,
            blur_score=blur_score,
            brightness=brightness,
            is_valid=is_valid,
            rejection_reason=rejection_reason
        )
    
    def _crop_face(self, frame: np.ndarray, bbox: np.ndarray, margin: float = 0.2) -> np.ndarray:
        """
        Crop face from frame with margin.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            margin: Margin ratio to add around face
            
        Returns:
            Cropped face image
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox[:4].astype(int)
        
        # Add margin
        face_w = x2 - x1
        face_h = y2 - y1
        margin_w = int(face_w * margin)
        margin_h = int(face_h * margin)
        
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        
        return frame[y1:y2, x1:x2].copy()
    
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect face in frame.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            DetectionResult with face crop and status
        """
        self._initialize()
        
        # Run detection
        faces = self._detector.get(frame)
        
        # Check number of faces
        if len(faces) == 0:
            return DetectionResult(status=DetectionStatus.NO_FACE)
        
        if len(faces) > 1:
            return DetectionResult(status=DetectionStatus.MULTI_FACE)
        
        # Single face found
        face = faces[0]
        bbox = face.bbox
        landmarks = face.kps if hasattr(face, 'kps') else None
        
        # Crop face
        face_crop = self._crop_face(frame, bbox)
        
        # Check quality
        quality = self._check_quality(face_crop, bbox)
        
        if not quality.is_valid:
            return DetectionResult(
                status=DetectionStatus.LOW_QUALITY,
                face_crop=face_crop,
                bounding_box=bbox,
                landmarks=landmarks,
                quality_metrics=quality
            )
        
        # Get aligned face for recognition (112x112)
        aligned_face = face.normed_embedding if hasattr(face, 'normed_embedding') else None
        
        return DetectionResult(
            status=DetectionStatus.OK,
            face_crop=face_crop,
            bounding_box=bbox,
            landmarks=landmarks,
            quality_metrics=quality,
            aligned_face=aligned_face
        )
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[DetectionResult]:
        """
        Detect faces in multiple frames.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            List of DetectionResult
        """
        return [self.detect(frame) for frame in frames]
    
    def get_face_crop_for_antispoofing(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        size: Tuple[int, int] = (80, 80)
    ) -> np.ndarray:
        """
        Get face crop sized for anti-spoofing model.
        
        Args:
            frame: Input frame
            bbox: Bounding box
            size: Target size for anti-spoofing
            
        Returns:
            Resized face crop
        """
        face_crop = self._crop_face(frame, bbox, margin=0.3)
        return cv2.resize(face_crop, size)


# Singleton instance
_detector_instance: Optional[FaceDetector] = None


def get_face_detector(
    min_face_size: int = 80,
    blur_threshold: float = 100.0,
    min_brightness: int = 40,
    max_brightness: int = 220
) -> FaceDetector:
    """Get or create face detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector(
            min_face_size=min_face_size,
            blur_threshold=blur_threshold,
            min_brightness=min_brightness,
            max_brightness=max_brightness
        )
    return _detector_instance
