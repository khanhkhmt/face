"""
Attendance Service for Face Recognition Attendance System.
Handles attendance checking with SERIAL and PARALLEL modes.
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import logging

from modules.face_detector import FaceDetector, DetectionStatus, DetectionResult, get_face_detector
from modules.anti_spoofing import FaceAntiSpoofing, AntiSpoofResult, get_anti_spoofing
from modules.face_recognition import FaceRecognition, MatchResult, get_face_recognition
from modules.score_fusion import ScoreFusion, FusionMethod, FusionResult, get_score_fusion
from config import config, PipelineMode


class AttendanceDecision(str, Enum):
    """Attendance decision."""
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class AttendanceReason(str, Enum):
    """Reason for attendance decision."""
    MATCHED = "MATCHED"
    SPOOF = "SPOOF"
    NON_MATCH = "NON_MATCH"
    NO_FACE = "NO_FACE"
    MULTI_FACE = "MULTI_FACE"
    LOW_QUALITY = "LOW_QUALITY"
    DUPLICATE = "DUPLICATE"


@dataclass
class AttendanceResult:
    """Standardized attendance result."""
    decision: str  # ACCEPT or REJECT
    reason: str  # MATCHED, SPOOF, NON_MATCH, etc.
    user_id: Optional[str]
    user_name: Optional[str]
    user_code: Optional[str]
    score_fas: Optional[float]
    score_fr: Optional[float]
    score_final: Optional[float]
    timestamp: str  # ISO8601 format
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class AttendanceService:
    """
    Attendance Service supporting SERIAL and PARALLEL modes.
    
    SERIAL Mode:
    1. Face Detection → 2. Anti-Spoofing → 3. Face Recognition
    
    PARALLEL Mode:
    1. Face Detection → 2a. Anti-Spoofing (parallel) 2b. Face Recognition → 3. Score Fusion
    """
    
    def __init__(
        self,
        mode: PipelineMode = PipelineMode.SERIAL,
        face_detector: Optional[FaceDetector] = None,
        anti_spoofing: Optional[FaceAntiSpoofing] = None,
        face_recognition: Optional[FaceRecognition] = None,
        score_fusion: Optional[ScoreFusion] = None,
        temporal_frames: int = 10,
        duplicate_window_minutes: int = 5
    ):
        """
        Initialize attendance service.
        
        Args:
            mode: Pipeline mode (SERIAL or PARALLEL)
            face_detector: Face detector instance
            anti_spoofing: Anti-spoofing instance
            face_recognition: Face recognition instance
            score_fusion: Score fusion instance
            temporal_frames: Number of frames for temporal smoothing
            duplicate_window_minutes: Duplicate check window
        """
        self.mode = mode
        self.face_detector = face_detector or get_face_detector(
            min_face_size=config.MIN_FACE_SIZE,
            blur_threshold=config.MAX_BLUR_THRESHOLD,
            min_brightness=config.MIN_BRIGHTNESS,
            max_brightness=config.MAX_BRIGHTNESS
        )
        self.anti_spoofing = anti_spoofing or get_anti_spoofing(threshold=config.T_FAS)
        self.face_recognition = face_recognition or get_face_recognition(threshold=config.T_FR)
        self.score_fusion = score_fusion or get_score_fusion(
            method=FusionMethod(config.FUSION_METHOD),
            t_fas=config.T_FAS,
            t_fr=config.T_FR,
            t_final=config.T_FINAL,
            t_fas_min=config.T_FAS_MIN,
            w1=config.W1,
            w2=config.W2
        )
        
        self.temporal_frames = temporal_frames
        self.duplicate_window_minutes = duplicate_window_minutes
    
    def _create_result(
        self,
        decision: AttendanceDecision,
        reason: AttendanceReason,
        user_id: Optional[str] = None,
        user_name: Optional[str] = None,
        user_code: Optional[str] = None,
        score_fas: Optional[float] = None,
        score_fr: Optional[float] = None,
        score_final: Optional[float] = None
    ) -> AttendanceResult:
        """Create standardized attendance result."""
        return AttendanceResult(
            decision=decision.value,
            reason=reason.value,
            user_id=user_id,
            user_name=user_name,
            user_code=user_code,
            score_fas=round(score_fas, 4) if score_fas is not None else None,
            score_fr=round(score_fr, 4) if score_fr is not None else None,
            score_final=round(score_final, 4) if score_final is not None else None,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    def _process_frames_detection(
        self,
        frames: List[np.ndarray]
    ) -> Tuple[List[DetectionResult], Optional[AttendanceReason]]:
        """
        Process frames through face detection.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            Tuple of (detection_results, error_reason)
        """
        detection_results = []
        
        for frame in frames:
            det_result = self.face_detector.detect(frame)
            
            # Any detection failure stops processing
            if det_result.status == DetectionStatus.NO_FACE:
                return [], AttendanceReason.NO_FACE
            elif det_result.status == DetectionStatus.MULTI_FACE:
                return [], AttendanceReason.MULTI_FACE
            elif det_result.status == DetectionStatus.LOW_QUALITY:
                return [], AttendanceReason.LOW_QUALITY
            
            detection_results.append(det_result)
        
        return detection_results, None
    
    def process_serial(
        self,
        frames: List[np.ndarray]
    ) -> AttendanceResult:
        """
        Process attendance using SERIAL mode.
        
        Pipeline: Detection → Anti-Spoofing → Recognition
        
        Args:
            frames: List of BGR frames
            
        Returns:
            AttendanceResult
        """
        # Step 1: Face Detection
        detection_results, error_reason = self._process_frames_detection(frames)
        
        if error_reason:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=error_reason
            )
        
        if len(detection_results) == 0:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=AttendanceReason.NO_FACE
            )
        
        # Step 2: Anti-Spoofing with temporal smoothing
        face_crops = [det.face_crop for det in detection_results]
        fas_result = self.anti_spoofing.check_with_temporal_smoothing(face_crops)
        
        if not fas_result.is_real:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=AttendanceReason.SPOOF,
                score_fas=fas_result.score
            )
        
        # Step 3: Face Recognition (use last frame for best quality)
        match_result = self.face_recognition.match_from_frame(frames[-1])
        
        if match_result.matched:
            return self._create_result(
                decision=AttendanceDecision.ACCEPT,
                reason=AttendanceReason.MATCHED,
                user_id=match_result.user_id,
                user_name=match_result.user_name,
                user_code=match_result.user_code,
                score_fas=fas_result.score,
                score_fr=match_result.score
            )
        else:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=AttendanceReason.NON_MATCH,
                score_fas=fas_result.score,
                score_fr=match_result.score
            )
    
    def process_parallel(
        self,
        frames: List[np.ndarray]
    ) -> AttendanceResult:
        """
        Process attendance using PARALLEL mode.
        
        Pipeline: Detection → [Anti-Spoofing || Recognition] → Score Fusion
        
        Args:
            frames: List of BGR frames
            
        Returns:
            AttendanceResult
        """
        # Step 1: Face Detection
        detection_results, error_reason = self._process_frames_detection(frames)
        
        if error_reason:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=error_reason
            )
        
        if len(detection_results) == 0:
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=AttendanceReason.NO_FACE
            )
        
        # Step 2: Parallel processing
        # 2a: Anti-Spoofing
        face_crops = [det.face_crop for det in detection_results]
        fas_result = self.anti_spoofing.check_with_temporal_smoothing(face_crops)
        
        # 2b: Face Recognition
        match_result = self.face_recognition.match_from_frame(frames[-1])
        
        # Step 3: Score Fusion
        fusion_result = self.score_fusion.fuse(
            score_fr=match_result.score,
            score_fas=fas_result.score
        )
        
        if fusion_result.accept:
            return self._create_result(
                decision=AttendanceDecision.ACCEPT,
                reason=AttendanceReason.MATCHED,
                user_id=match_result.user_id,
                user_name=match_result.user_name,
                user_code=match_result.user_code,
                score_fas=fas_result.score,
                score_fr=match_result.score,
                score_final=fusion_result.score_final
            )
        else:
            # Map fusion reason to attendance reason
            if fusion_result.reason == "SPOOF":
                reason = AttendanceReason.SPOOF
            elif fusion_result.reason == "NON_MATCH":
                reason = AttendanceReason.NON_MATCH
            else:
                reason = AttendanceReason.NON_MATCH
            
            return self._create_result(
                decision=AttendanceDecision.REJECT,
                reason=reason,
                score_fas=fas_result.score,
                score_fr=match_result.score,
                score_final=fusion_result.score_final
            )
    
    def process(
        self,
        frames: List[np.ndarray]
    ) -> AttendanceResult:
        """
        Process attendance using configured mode.
        
        Args:
            frames: List of BGR frames
            
        Returns:
            AttendanceResult
        """
        if self.mode == PipelineMode.SERIAL:
            return self.process_serial(frames)
        else:
            return self.process_parallel(frames)
    
    async def process_with_duplicate_check(
        self,
        frames: List[np.ndarray],
        session,  # AsyncSession
        save_snapshot: bool = False,
        snapshot_dir: str = "./snapshots"
    ) -> AttendanceResult:
        """
        Process attendance with duplicate check and logging.
        
        Args:
            frames: List of BGR frames
            session: Database session
            save_snapshot: Whether to save face snapshot
            snapshot_dir: Directory for snapshots
            
        Returns:
            AttendanceResult
        """
        from database import crud
        import os
        
        # Process attendance
        result = self.process(frames)
        
        # Check for duplicate if accepted
        if result.decision == AttendanceDecision.ACCEPT.value and result.user_id:
            is_duplicate = await crud.check_duplicate_attendance(
                session,
                result.user_id,
                self.duplicate_window_minutes
            )
            
            if is_duplicate:
                result = self._create_result(
                    decision=AttendanceDecision.REJECT,
                    reason=AttendanceReason.DUPLICATE,
                    user_id=result.user_id,
                    user_name=result.user_name,
                    user_code=result.user_code,
                    score_fas=result.score_fas,
                    score_fr=result.score_fr,
                    score_final=result.score_final
                )
        
        # Save snapshot if requested
        snapshot_path = None
        if save_snapshot and len(frames) > 0:
            os.makedirs(snapshot_dir, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            snapshot_path = os.path.join(snapshot_dir, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(snapshot_path, frames[-1])
        
        # Log attendance
        await crud.create_attendance_log(
            session,
            user_id=result.user_id,
            decision=result.decision,
            reason=result.reason,
            score_fas=result.score_fas,
            score_fr=result.score_fr,
            score_final=result.score_final,
            snapshot_path=snapshot_path
        )
        
        return result
    
    def set_mode(self, mode: PipelineMode):
        """Change pipeline mode."""
        self.mode = mode
        logging.info(f"Attendance mode changed to {mode.value}")
    
    def update_thresholds(
        self,
        t_fas: Optional[float] = None,
        t_fr: Optional[float] = None,
        t_final: Optional[float] = None
    ):
        """Update thresholds."""
        if t_fas is not None:
            self.anti_spoofing.threshold = t_fas
            self.score_fusion.t_fas = t_fas
        if t_fr is not None:
            self.face_recognition.threshold = t_fr
            self.score_fusion.t_fr = t_fr
        if t_final is not None:
            self.score_fusion.t_final = t_final


# Frame buffer for collecting temporal frames
class FrameBuffer:
    """
    Buffer for collecting frames for temporal smoothing.
    """
    
    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.frames: List[np.ndarray] = []
    
    def add(self, frame: np.ndarray):
        """Add frame to buffer."""
        self.frames.append(frame)
        if len(self.frames) > self.max_size:
            self.frames.pop(0)
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self.frames) >= self.max_size
    
    def get_frames(self) -> List[np.ndarray]:
        """Get all frames."""
        return self.frames.copy()
    
    def clear(self):
        """Clear buffer."""
        self.frames.clear()
    
    def __len__(self) -> int:
        return len(self.frames)


# Singleton instance
_attendance_service: Optional[AttendanceService] = None


def get_attendance_service() -> AttendanceService:
    """Get or create attendance service singleton."""
    global _attendance_service
    if _attendance_service is None:
        _attendance_service = AttendanceService(
            mode=PipelineMode(config.MODE),
            temporal_frames=config.TEMPORAL_FRAMES,
            duplicate_window_minutes=config.DUPLICATE_WINDOW_MINUTES
        )
    return _attendance_service
