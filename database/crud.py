"""
CRUD operations for Face Recognition Attendance System.
"""
from datetime import datetime, timedelta
from typing import Optional, List, Tuple
import numpy as np
from sqlalchemy import select, delete, and_
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from .models import User, FaceTemplate, AttendanceLog


# ==================== User Operations ====================

async def create_user(
    session: AsyncSession,
    name: str,
    code: str
) -> User:
    """Create a new user."""
    user = User(name=name, code=code)
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def get_user_by_id(session: AsyncSession, user_id: str) -> Optional[User]:
    """Get user by ID."""
    result = await session.execute(select(User).where(User.user_id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_code(session: AsyncSession, code: str) -> Optional[User]:
    """Get user by code."""
    result = await session.execute(select(User).where(User.code == code))
    return result.scalar_one_or_none()


async def get_all_users(session: AsyncSession) -> List[User]:
    """Get all users."""
    result = await session.execute(select(User).order_by(User.created_at.desc()))
    return list(result.scalars().all())


async def delete_user(session: AsyncSession, user_id: str) -> bool:
    """Delete a user and all related templates."""
    user = await get_user_by_id(session, user_id)
    if user:
        await session.delete(user)
        await session.commit()
        return True
    return False


# ==================== Face Template Operations ====================

async def create_face_template(
    session: AsyncSession,
    user_id: str,
    embedding: np.ndarray
) -> FaceTemplate:
    """Create a new face template for a user."""
    # Convert numpy array to bytes
    embedding_bytes = embedding.tobytes()
    template = FaceTemplate(user_id=user_id, embedding=embedding_bytes)
    session.add(template)
    await session.commit()
    await session.refresh(template)
    return template


async def create_face_templates_batch(
    session: AsyncSession,
    user_id: str,
    embeddings: List[np.ndarray]
) -> List[FaceTemplate]:
    """Create multiple face templates for a user."""
    templates = []
    for embedding in embeddings:
        embedding_bytes = embedding.tobytes()
        template = FaceTemplate(user_id=user_id, embedding=embedding_bytes)
        session.add(template)
        templates.append(template)
    await session.commit()
    for template in templates:
        await session.refresh(template)
    return templates


async def get_user_templates(
    session: AsyncSession,
    user_id: str
) -> List[FaceTemplate]:
    """Get all face templates for a user."""
    result = await session.execute(
        select(FaceTemplate).where(FaceTemplate.user_id == user_id)
    )
    return list(result.scalars().all())


async def get_all_templates_with_users(
    session: AsyncSession
) -> List[Tuple[User, FaceTemplate]]:
    """Get all templates with their associated users."""
    result = await session.execute(
        select(User, FaceTemplate).join(FaceTemplate)
    )
    return list(result.all())


def bytes_to_embedding(embedding_bytes: bytes, dtype=np.float32, dim: int = 512) -> np.ndarray:
    """Convert bytes back to numpy array."""
    return np.frombuffer(embedding_bytes, dtype=dtype).reshape(dim)


async def delete_user_templates(session: AsyncSession, user_id: str) -> int:
    """Delete all face templates for a user."""
    result = await session.execute(
        delete(FaceTemplate).where(FaceTemplate.user_id == user_id)
    )
    await session.commit()
    return result.rowcount


# ==================== Attendance Log Operations ====================

async def create_attendance_log(
    session: AsyncSession,
    user_id: Optional[str],
    decision: str,
    reason: str,
    score_fas: Optional[float] = None,
    score_fr: Optional[float] = None,
    score_final: Optional[float] = None,
    snapshot_path: Optional[str] = None
) -> AttendanceLog:
    """Create a new attendance log entry."""
    log = AttendanceLog(
        user_id=user_id,
        decision=decision,
        reason=reason,
        score_fas=score_fas,
        score_fr=score_fr,
        score_final=score_final,
        snapshot_path=snapshot_path
    )
    session.add(log)
    await session.commit()
    await session.refresh(log)
    return log


async def get_attendance_logs(
    session: AsyncSession,
    limit: int = 100,
    user_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[AttendanceLog]:
    """Get attendance logs with optional filters."""
    query = select(AttendanceLog)
    
    conditions = []
    if user_id:
        conditions.append(AttendanceLog.user_id == user_id)
    if start_date:
        conditions.append(AttendanceLog.timestamp >= start_date)
    if end_date:
        conditions.append(AttendanceLog.timestamp <= end_date)
    
    if conditions:
        query = query.where(and_(*conditions))
    
    query = query.order_by(AttendanceLog.timestamp.desc()).limit(limit)
    result = await session.execute(query)
    return list(result.scalars().all())


async def check_duplicate_attendance(
    session: AsyncSession,
    user_id: str,
    window_minutes: int
) -> bool:
    """Check if user has already checked in within the time window."""
    cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
    
    result = await session.execute(
        select(AttendanceLog).where(
            and_(
                AttendanceLog.user_id == user_id,
                AttendanceLog.decision == "ACCEPT",
                AttendanceLog.timestamp >= cutoff_time
            )
        ).limit(1)
    )
    
    return result.scalar_one_or_none() is not None


async def get_latest_attendance(
    session: AsyncSession,
    user_id: str
) -> Optional[AttendanceLog]:
    """Get the latest attendance log for a user."""
    result = await session.execute(
        select(AttendanceLog).where(
            AttendanceLog.user_id == user_id
        ).order_by(AttendanceLog.timestamp.desc()).limit(1)
    )
    return result.scalar_one_or_none()


# ==================== Synchronous Operations (for initialization) ====================

def create_user_sync(session: Session, name: str, code: str) -> User:
    """Create a new user (synchronous)."""
    user = User(name=name, code=code)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def create_face_template_sync(
    session: Session,
    user_id: str,
    embedding: np.ndarray
) -> FaceTemplate:
    """Create a new face template (synchronous)."""
    embedding_bytes = embedding.tobytes()
    template = FaceTemplate(user_id=user_id, embedding=embedding_bytes)
    session.add(template)
    session.commit()
    session.refresh(template)
    return template


def get_all_templates_with_users_sync(session: Session) -> List[Tuple[User, FaceTemplate]]:
    """Get all templates with their associated users (synchronous)."""
    result = session.execute(select(User, FaceTemplate).join(FaceTemplate))
    return list(result.all())
