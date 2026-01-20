"""
Database models for Face Recognition Attendance System.
Tables: Users, FaceTemplates, AttendanceLogs
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, String, Integer, Float, DateTime, 
    ForeignKey, LargeBinary, Text, create_engine
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker as async_sessionmaker
import uuid

Base = declarative_base()


def generate_uuid() -> str:
    """Generate a unique ID."""
    return str(uuid.uuid4())


class User(Base):
    """User table for enrolled users."""
    __tablename__ = "users"
    
    user_id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    code = Column(String(50), unique=True, nullable=False)  # Employee/Student code
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    face_templates = relationship("FaceTemplate", back_populates="user", cascade="all, delete-orphan")
    attendance_logs = relationship("AttendanceLog", back_populates="user")
    
    def __repr__(self):
        return f"<User(user_id={self.user_id}, name={self.name}, code={self.code})>"


class FaceTemplate(Base):
    """Face template (embedding) storage for enrolled users."""
    __tablename__ = "face_templates"
    
    template_id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False)
    embedding = Column(LargeBinary, nullable=False)  # Stored as binary (numpy array bytes)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="face_templates")
    
    def __repr__(self):
        return f"<FaceTemplate(template_id={self.template_id}, user_id={self.user_id})>"


class AttendanceLog(Base):
    """Attendance log for all attendance attempts."""
    __tablename__ = "attendance_logs"
    
    log_id = Column(String(36), primary_key=True, default=generate_uuid)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=True)  # Nullable for failed attempts
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    decision = Column(String(10), nullable=False)  # ACCEPT or REJECT
    reason = Column(String(20), nullable=False)  # MATCHED, SPOOF, NON_MATCH, etc.
    score_fas = Column(Float, nullable=True)
    score_fr = Column(Float, nullable=True)
    score_final = Column(Float, nullable=True)  # Only for PARALLEL mode
    snapshot_path = Column(Text, nullable=True)  # Path to captured frame
    
    # Relationships
    user = relationship("User", back_populates="attendance_logs")
    
    def __repr__(self):
        return f"<AttendanceLog(log_id={self.log_id}, decision={self.decision}, reason={self.reason})>"


# Database engine and session setup
def get_engine(database_url: str):
    """Create database engine."""
    # For SQLite, use aiosqlite for async support
    if database_url.startswith("sqlite:"):
        # Convert to async URL
        async_url = database_url.replace("sqlite:", "sqlite+aiosqlite:")
        return create_async_engine(async_url, echo=False)
    return create_async_engine(database_url, echo=False)


def get_sync_engine(database_url: str):
    """Create synchronous database engine for initialization."""
    return create_engine(database_url, echo=False)


async def create_tables(engine):
    """Create all tables in the database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def create_tables_sync(database_url: str):
    """Create all tables synchronously."""
    engine = get_sync_engine(database_url)
    Base.metadata.create_all(engine)
    return engine
