"""
Configuration management for Face Recognition Attendance System.
All thresholds and parameters can be configured here.
"""
from enum import Enum
from typing import Literal
from pydantic import BaseModel
import os


class PipelineMode(str, Enum):
    SERIAL = "SERIAL"
    PARALLEL = "PARALLEL"


class FusionMethod(str, Enum):
    AND_GATE = "AND_GATE"
    WEIGHTED = "WEIGHTED"


class Config(BaseModel):
    """Main configuration for the attendance system."""
    
    # Pipeline mode: SERIAL or PARALLEL
    MODE: PipelineMode = PipelineMode.SERIAL
    
    # Fusion method for PARALLEL mode
    FUSION_METHOD: FusionMethod = FusionMethod.AND_GATE
    
    # Anti-Spoofing threshold
    T_FAS: float = 0.5
    
    # Face Recognition threshold (cosine similarity)
    T_FR: float = 0.4
    
    # Final score threshold for PARALLEL weighted fusion
    T_FINAL: float = 0.6
    
    # Minimum FAS score for PARALLEL mode (even with weighted fusion)
    T_FAS_MIN: float = 0.3
    
    # Weights for weighted score fusion
    W1: float = 0.6  # Weight for FR score
    W2: float = 0.4  # Weight for FAS score
    
    # Number of frames for temporal smoothing in anti-spoofing
    TEMPORAL_FRAMES: int = 10
    
    # Duplicate check window in minutes
    DUPLICATE_WINDOW_MINUTES: int = 5
    
    # Minimum number of valid embeddings required for enrollment
    MIN_ENROLL_EMBEDDINGS: int = 5
    
    # Maximum number of embeddings to store per user
    MAX_ENROLL_EMBEDDINGS: int = 10
    
    # Face quality thresholds
    MIN_FACE_SIZE: int = 80  # Minimum face size in pixels
    MAX_BLUR_THRESHOLD: float = 100.0  # Laplacian variance threshold
    MIN_BRIGHTNESS: int = 40
    MAX_BRIGHTNESS: int = 220
    
    # Enrollment video duration in seconds
    ENROLL_DURATION: float = 3.0
    
    # Database path
    DATABASE_URL: str = "sqlite:///./attendance.db"
    
    # Snapshot storage path
    SNAPSHOT_DIR: str = "./snapshots"
    
    # Model paths
    INSIGHTFACE_MODEL: str = "buffalo_l"
    
    class Config:
        use_enum_values = True


# Load configuration from environment or use defaults
def load_config() -> Config:
    """Load configuration from environment variables or use defaults."""
    config = Config()
    
    # Override from environment if available
    if os.getenv("ATTENDANCE_MODE"):
        config.MODE = PipelineMode(os.getenv("ATTENDANCE_MODE"))
    
    if os.getenv("ATTENDANCE_FUSION_METHOD"):
        config.FUSION_METHOD = FusionMethod(os.getenv("ATTENDANCE_FUSION_METHOD"))
    
    if os.getenv("ATTENDANCE_T_FAS"):
        config.T_FAS = float(os.getenv("ATTENDANCE_T_FAS"))
    
    if os.getenv("ATTENDANCE_T_FR"):
        config.T_FR = float(os.getenv("ATTENDANCE_T_FR"))
    
    if os.getenv("ATTENDANCE_T_FINAL"):
        config.T_FINAL = float(os.getenv("ATTENDANCE_T_FINAL"))
    
    if os.getenv("ATTENDANCE_DUPLICATE_WINDOW"):
        config.DUPLICATE_WINDOW_MINUTES = int(os.getenv("ATTENDANCE_DUPLICATE_WINDOW"))
    
    if os.getenv("DATABASE_URL"):
        config.DATABASE_URL = os.getenv("DATABASE_URL")
    
    return config


# Global configuration instance
config = load_config()
