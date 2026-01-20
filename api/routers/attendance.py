"""
Attendance API Router.
Handles attendance checking with SERIAL and PARALLEL modes.
"""
import cv2
import numpy as np
import tempfile
import os
import base64
import json
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    AttendanceResponse, AttendanceLogResponse, AttendanceLogsResponse
)
from database.session import get_async_session, get_async_session_context
from database import crud
from services.attendance_service import get_attendance_service, FrameBuffer, AttendanceResult
from config import config


router = APIRouter(prefix="/api", tags=["attendance"])


@router.post("/attendance", response_model=AttendanceResponse)
async def check_attendance_video(
    video: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Check attendance with video upload.
    
    The video should be 1-3 seconds of the user's face.
    The system will:
    1. Extract frames for temporal smoothing
    2. Detect face and check quality
    3. Check anti-spoofing
    4. Match against enrolled users
    
    Args:
        video: Video file (mp4, webm, etc.)
    
    Returns:
        AttendanceResponse with decision and scores
    """
    # Validate file type
    if not video.content_type.startswith("video/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be a video"
        )
    
    # Save video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract frames
        frames = []
        cap = cv2.VideoCapture(tmp_path)
        frame_count = 0
        
        while cap.isOpened() and len(frames) < config.TEMPORAL_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 3 == 0:  # Skip every 3rd frame
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        
        if len(frames) == 0:
            return AttendanceResponse(
                decision="REJECT",
                reason="NO_FACE",
                timestamp=datetime.utcnow().isoformat() + "Z"
            )
        
        # Process attendance
        attendance_service = get_attendance_service()
        result = await attendance_service.process_with_duplicate_check(
            frames=frames,
            session=session,
            save_snapshot=True,
            snapshot_dir=config.SNAPSHOT_DIR
        )
        
        return AttendanceResponse(
            decision=result.decision,
            reason=result.reason,
            user_id=result.user_id,
            user_name=result.user_name,
            user_code=result.user_code,
            score_fas=result.score_fas,
            score_fr=result.score_fr,
            score_final=result.score_final,
            timestamp=result.timestamp
        )
        
    finally:
        os.unlink(tmp_path)


@router.post("/attendance/image", response_model=AttendanceResponse)
async def check_attendance_image(
    image: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Check attendance with single image.
    
    Note: Single image provides less anti-spoofing accuracy than video.
    For better security, use video endpoint or WebSocket for real-time.
    
    Args:
        image: Face image (jpg, png, etc.)
    
    Returns:
        AttendanceResponse with decision and scores
    """
    # Validate file type
    if not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be an image"
        )
    
    # Decode image
    content = await image.read()
    nparr = np.frombuffer(content, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return AttendanceResponse(
            decision="REJECT",
            reason="NO_FACE",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    # Use single frame (less accurate anti-spoofing)
    frames = [frame]
    
    # Process attendance
    attendance_service = get_attendance_service()
    result = await attendance_service.process_with_duplicate_check(
        frames=frames,
        session=session,
        save_snapshot=True,
        snapshot_dir=config.SNAPSHOT_DIR
    )
    
    return AttendanceResponse(
        decision=result.decision,
        reason=result.reason,
        user_id=result.user_id,
        user_name=result.user_name,
        user_code=result.user_code,
        score_fas=result.score_fas,
        score_fr=result.score_fr,
        score_final=result.score_final,
        timestamp=result.timestamp
    )


@router.post("/attendance/frames", response_model=AttendanceResponse)
async def check_attendance_frames(
    images: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Check attendance with multiple image frames.
    
    Provides temporal smoothing like video but with explicit frames.
    Send 5-10 consecutive frames for best results.
    
    Args:
        images: List of face images
    
    Returns:
        AttendanceResponse with decision and scores
    """
    if len(images) < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 1 image is required"
        )
    
    # Decode images
    frames = []
    for img_file in images:
        content = await img_file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    
    if len(frames) == 0:
        return AttendanceResponse(
            decision="REJECT",
            reason="NO_FACE",
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
    
    # Process attendance
    attendance_service = get_attendance_service()
    result = await attendance_service.process_with_duplicate_check(
        frames=frames,
        session=session,
        save_snapshot=True,
        snapshot_dir=config.SNAPSHOT_DIR
    )
    
    return AttendanceResponse(
        decision=result.decision,
        reason=result.reason,
        user_id=result.user_id,
        user_name=result.user_name,
        user_code=result.user_code,
        score_fas=result.score_fas,
        score_fr=result.score_fr,
        score_final=result.score_final,
        timestamp=result.timestamp
    )


@router.get("/attendance/logs", response_model=AttendanceLogsResponse)
async def get_attendance_logs(
    limit: int = Query(100, ge=1, le=1000),
    user_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get attendance logs with optional filters.
    
    Args:
        limit: Maximum number of logs to return
        user_id: Filter by user ID
        start_date: Filter by start date
        end_date: Filter by end date
    
    Returns:
        AttendanceLogsResponse with logs
    """
    logs = await crud.get_attendance_logs(
        session,
        limit=limit,
        user_id=user_id,
        start_date=start_date,
        end_date=end_date
    )
    
    log_responses = []
    for log in logs:
        user = await crud.get_user_by_id(session, log.user_id) if log.user_id else None
        log_responses.append(AttendanceLogResponse(
            log_id=log.log_id,
            user_id=log.user_id,
            user_name=user.name if user else None,
            user_code=user.code if user else None,
            timestamp=log.timestamp,
            decision=log.decision,
            reason=log.reason,
            score_fas=log.score_fas,
            score_fr=log.score_fr,
            score_final=log.score_final,
            snapshot_path=log.snapshot_path
        ))
    
    return AttendanceLogsResponse(
        logs=log_responses,
        total=len(log_responses)
    )


@router.get("/attendance/today")
async def get_today_attendance(
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get all attendance for today.
    
    Returns:
        Summary of today's attendance
    """
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    today_end = today_start + timedelta(days=1)
    
    logs = await crud.get_attendance_logs(
        session,
        limit=1000,
        start_date=today_start,
        end_date=today_end
    )
    
    # Count by decision
    accepted = sum(1 for log in logs if log.decision == "ACCEPT")
    rejected = sum(1 for log in logs if log.decision == "REJECT")
    
    # Get unique users
    unique_users = set(log.user_id for log in logs if log.user_id and log.decision == "ACCEPT")
    
    return {
        "date": today_start.date().isoformat(),
        "total_attempts": len(logs),
        "accepted": accepted,
        "rejected": rejected,
        "unique_users": len(unique_users)
    }


# WebSocket for real-time attendance
@router.websocket("/ws/attendance")
async def websocket_attendance(websocket: WebSocket):
    """
    WebSocket endpoint for real-time attendance.
    
    Protocol:
    1. Client connects
    2. Client sends base64-encoded frames as JSON: {"type": "frame", "data": "<base64>"}
    3. Server buffers frames until TEMPORAL_FRAMES reached
    4. Server processes and sends result: {"type": "result", "data": {...}}
    5. Client can send new frames for next attendance
    
    To reset buffer, send: {"type": "reset"}
    """
    await websocket.accept()
    
    frame_buffer = FrameBuffer(max_size=config.TEMPORAL_FRAMES)
    attendance_service = get_attendance_service()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get("type", "")
            
            if msg_type == "reset":
                frame_buffer.clear()
                await websocket.send_json({
                    "type": "status",
                    "message": "Buffer reset"
                })
                continue
            
            if msg_type == "frame":
                # Decode base64 frame
                try:
                    frame_data = message.get("data", "")
                    frame_bytes = base64.b64decode(frame_data)
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid frame data"
                        })
                        continue
                    
                    # Add to buffer
                    frame_buffer.add(frame)
                    
                    # Send status
                    await websocket.send_json({
                        "type": "status",
                        "message": f"Frame {len(frame_buffer)}/{config.TEMPORAL_FRAMES}",
                        "frames_collected": len(frame_buffer),
                        "frames_needed": config.TEMPORAL_FRAMES
                    })
                    
                    # Process if buffer is full
                    if frame_buffer.is_full():
                        async with get_async_session_context() as session:
                            result = await attendance_service.process_with_duplicate_check(
                                frames=frame_buffer.get_frames(),
                                session=session,
                                save_snapshot=True,
                                snapshot_dir=config.SNAPSHOT_DIR
                            )
                        
                        await websocket.send_json({
                            "type": "result",
                            "data": result.to_dict()
                        })
                        
                        # Clear buffer for next attendance
                        frame_buffer.clear()
                        
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
            
            elif msg_type == "process":
                # Process immediately with current frames
                if len(frame_buffer) == 0:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No frames in buffer"
                    })
                    continue
                
                async with get_async_session_context() as session:
                    result = await attendance_service.process_with_duplicate_check(
                        frames=frame_buffer.get_frames(),
                        session=session,
                        save_snapshot=True,
                        snapshot_dir=config.SNAPSHOT_DIR
                    )
                
                await websocket.send_json({
                    "type": "result",
                    "data": result.to_dict()
                })
                
                frame_buffer.clear()
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
