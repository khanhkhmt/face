"""
Face Anti-Spoofing Module.
Detects fake faces (printed photos, video replay, phone screens).
Uses a lightweight CNN-based approach similar to Silent-Face-Anti-Spoofing.
"""
import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import logging
import os

try:
    import onnxruntime as ort
except ImportError:
    ort = None
    logging.warning("ONNX Runtime not installed. Anti-spoofing may not work.")


@dataclass
class AntiSpoofResult:
    """Result of anti-spoofing check."""
    score: float  # 0.0 (fake) to 1.0 (real)
    is_real: bool
    label: str  # "REAL" or "FAKE"


class FaceAntiSpoofing:
    """
    Face Anti-Spoofing module.
    
    Uses texture analysis and CNN-based classification to detect:
    - Printed photo attacks
    - Video replay attacks
    - Screen display attacks
    
    Supports temporal smoothing for more robust decisions.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        model_path: Optional[str] = None
    ):
        """
        Initialize anti-spoofing module.
        
        Args:
            threshold: Classification threshold (>= threshold means real)
            model_path: Path to ONNX model (optional, uses texture analysis if not provided)
        """
        self.threshold = threshold
        self.model_path = model_path
        self._model = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization."""
        if self._initialized:
            return
        
        # Try to load ONNX model if available
        if self.model_path and os.path.exists(self.model_path) and ort:
            try:
                self._model = ort.InferenceSession(
                    self.model_path,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                logging.info(f"Anti-spoofing model loaded from {self.model_path}")
            except Exception as e:
                logging.warning(f"Failed to load anti-spoofing model: {e}")
                self._model = None
        
        self._initialized = True
    
    def _analyze_texture(self, face_image: np.ndarray) -> float:
        """
        Analyze face texture for spoofing detection.
        Uses LBP (Local Binary Pattern) and frequency analysis.
        
        Args:
            face_image: Face crop (BGR)
            
        Returns:
            Spoof score (higher = more likely real)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to standard size
        gray = cv2.resize(gray, (128, 128))
        
        # 1. Frequency analysis - real faces have more high-frequency content
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Calculate ratio of high frequency to total
        center = magnitude.shape[0] // 2
        radius = 20  # Low frequency radius
        
        # Create mask for low frequency
        y, x = np.ogrid[:magnitude.shape[0], :magnitude.shape[1]]
        low_freq_mask = (x - center) ** 2 + (y - center) ** 2 <= radius ** 2
        
        low_freq_energy = np.sum(magnitude[low_freq_mask])
        total_energy = np.sum(magnitude)
        
        if total_energy == 0:
            freq_ratio = 0.5
        else:
            freq_ratio = 1 - (low_freq_energy / total_energy)  # Higher = more high freq = more real
        
        # 2. Laplacian variance (sharpness) - real faces tend to have more texture
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        laplacian_score = min(laplacian_var / 500.0, 1.0)  # Normalize
        
        # 3. Color analysis - printed photos often have limited color range
        if len(face_image.shape) == 3:
            hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
            # Saturation variance - real faces have more varied saturation
            sat_var = np.var(hsv[:, :, 1])
            sat_score = min(sat_var / 1000.0, 1.0)
        else:
            sat_score = 0.5
        
        # 4. Edge density - moire patterns from screens/prints
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # Too many edges (moire) or too few (blurred) is suspicious
        edge_score = 1.0 - abs(edge_density - 0.1) * 5
        edge_score = max(0, min(1, edge_score))
        
        # 5. Reflection detection - screens often have reflections
        # Look for bright spots
        _, bright_mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
        reflection_score = 1.0 if bright_ratio < 0.05 else max(0, 1.0 - bright_ratio * 10)
        
        # Combine scores with weights
        combined_score = (
            freq_ratio * 0.25 +
            laplacian_score * 0.25 +
            sat_score * 0.20 +
            edge_score * 0.15 +
            reflection_score * 0.15
        )
        
        return float(combined_score)
    
    def _model_inference(self, face_image: np.ndarray) -> float:
        """
        Run model inference for anti-spoofing.
        
        Args:
            face_image: Face crop (BGR)
            
        Returns:
            Real probability score
        """
        if self._model is None:
            return self._analyze_texture(face_image)
        
        # Preprocess for model
        img = cv2.resize(face_image, (80, 80))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # CHW
        img = np.expand_dims(img, axis=0)  # NCHW
        
        # Run inference
        input_name = self._model.get_inputs()[0].name
        output = self._model.run(None, {input_name: img})
        
        # Get probability (assuming softmax output: [fake_prob, real_prob])
        if len(output[0].shape) > 1 and output[0].shape[1] >= 2:
            real_prob = float(output[0][0][1])
        else:
            real_prob = float(output[0][0])
        
        return real_prob
    
    def check(self, face_image: np.ndarray) -> AntiSpoofResult:
        """
        Check if face is real or fake.
        
        Args:
            face_image: Face crop (BGR)
            
        Returns:
            AntiSpoofResult with score and decision
        """
        self._initialize()
        
        # Get spoof score
        score = self._model_inference(face_image)
        
        # Make decision
        is_real = score >= self.threshold
        label = "REAL" if is_real else "FAKE"
        
        return AntiSpoofResult(
            score=score,
            is_real=is_real,
            label=label
        )
    
    def check_with_temporal_smoothing(
        self,
        face_images: List[np.ndarray],
        method: str = "median"
    ) -> AntiSpoofResult:
        """
        Check multiple frames with temporal smoothing.
        
        Args:
            face_images: List of face crops
            method: Smoothing method ("median", "mean", "min")
            
        Returns:
            AntiSpoofResult with smoothed score
        """
        if len(face_images) == 0:
            return AntiSpoofResult(score=0.0, is_real=False, label="FAKE")
        
        # Get scores for all frames
        scores = [self.check(img).score for img in face_images]
        
        # Apply smoothing
        if method == "median":
            final_score = float(np.median(scores))
        elif method == "mean":
            final_score = float(np.mean(scores))
        elif method == "min":
            final_score = float(np.min(scores))
        else:
            final_score = float(np.median(scores))
        
        is_real = final_score >= self.threshold
        label = "REAL" if is_real else "FAKE"
        
        return AntiSpoofResult(
            score=final_score,
            is_real=is_real,
            label=label
        )


# Singleton instance
_antispoof_instance: Optional[FaceAntiSpoofing] = None


def get_anti_spoofing(threshold: float = 0.5, model_path: Optional[str] = None) -> FaceAntiSpoofing:
    """Get or create anti-spoofing singleton."""
    global _antispoof_instance
    if _antispoof_instance is None:
        _antispoof_instance = FaceAntiSpoofing(threshold=threshold, model_path=model_path)
    return _antispoof_instance
