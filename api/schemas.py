"""
Pydantic schemas for API requests and responses.
"""
from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# ==================== User Schemas ====================

class UserCreate(BaseModel):
    """Schema for creating a new user."""
    name: str = Field(..., min_length=1, max_length=255, description="User's full name")
    code: str = Field(..., min_length=1, max_length=50, description="Employee/Student code")


class UserResponse(BaseModel):
    """Schema for user response."""
    user_id: str
    name: str
    code: str
    created_at: datetime
    embeddings_count: int = 0
    
    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    """Schema for user list response."""
    users: List[UserResponse]
    total: int


# ==================== Enrollment Schemas ====================

class EnrollRequest(BaseModel):
    """Schema for enrollment request (metadata only, video sent as file)."""
    name: str = Field(..., min_length=1, max_length=255)
    code: str = Field(..., min_length=1, max_length=50)


class EnrollResponse(BaseModel):
    """Schema for enrollment response."""
    success: bool
    user_id: Optional[str]
    message: str
    embeddings_count: int
    frames_processed: int
    valid_frames: int


# ==================== Attendance Schemas ====================

class AttendanceResponse(BaseModel):
    """Standardized attendance response."""
    decision: Literal["ACCEPT", "REJECT"]
    reason: Literal["MATCHED", "SPOOF", "NON_MATCH", "NO_FACE", 
                    "MULTI_FACE", "LOW_QUALITY", "DUPLICATE"]
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    user_code: Optional[str] = None
    score_fas: Optional[float] = None
    score_fr: Optional[float] = None
    score_final: Optional[float] = None
    timestamp: str = Field(..., description="ISO8601 timestamp")


class AttendanceLogResponse(BaseModel):
    """Schema for attendance log entry."""
    log_id: str
    user_id: Optional[str]
    user_name: Optional[str] = None
    user_code: Optional[str] = None
    timestamp: datetime
    decision: str
    reason: str
    score_fas: Optional[float]
    score_fr: Optional[float]
    score_final: Optional[float]
    snapshot_path: Optional[str]
    
    class Config:
        from_attributes = True


class AttendanceLogsResponse(BaseModel):
    """Schema for attendance logs list."""
    logs: List[AttendanceLogResponse]
    total: int


# ==================== Configuration Schemas ====================

class ConfigResponse(BaseModel):
    """Current configuration."""
    mode: str
    fusion_method: str
    t_fas: float
    t_fr: float
    t_final: float
    t_fas_min: float
    w1: float
    w2: float
    temporal_frames: int
    duplicate_window_minutes: int


class ConfigUpdate(BaseModel):
    """Schema for updating configuration."""
    mode: Optional[Literal["SERIAL", "PARALLEL"]] = None
    fusion_method: Optional[Literal["AND_GATE", "WEIGHTED"]] = None
    t_fas: Optional[float] = Field(None, ge=0, le=1)
    t_fr: Optional[float] = Field(None, ge=0, le=1)
    t_final: Optional[float] = Field(None, ge=0, le=1)
    t_fas_min: Optional[float] = Field(None, ge=0, le=1)
    w1: Optional[float] = Field(None, ge=0, le=1)
    w2: Optional[float] = Field(None, ge=0, le=1)
    temporal_frames: Optional[int] = Field(None, ge=1, le=30)
    duplicate_window_minutes: Optional[int] = Field(None, ge=1, le=1440)


# ==================== WebSocket Schemas ====================

class WSMessage(BaseModel):
    """WebSocket message."""
    type: str  # "frame", "result", "error", "status"
    data: Optional[dict] = None
    message: Optional[str] = None


# ==================== Health Check ====================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    mode: str
    registered_users: int
    database_connected: bool
