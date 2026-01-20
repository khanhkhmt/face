"""
Enrollment API Router.
Handles user registration with face capture.
"""
import cv2
import numpy as np
import tempfile
import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.schemas import (
    UserCreate, UserResponse, UserListResponse, 
    EnrollRequest, EnrollResponse
)
from database.session import get_async_session
from database import crud
from services.enroll_service import get_enrollment_service
from modules.face_recognition import get_face_recognition


router = APIRouter(prefix="/api", tags=["enrollment"])


@router.post("/enroll", response_model=EnrollResponse)
async def enroll_user(
    name: str = Form(...),
    code: str = Form(...),
    video: UploadFile = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Enroll a new user with video upload.
    
    The video should be 2-3 seconds of the user's face clearly visible.
    The system will:
    1. Extract frames from video
    2. Detect face and check quality
    3. Verify anti-spoofing (must be real face)
    4. Extract and store face embeddings
    
    Args:
        name: User's full name
        code: Employee/Student code (unique)
        video: Video file (mp4, webm, etc.)
    
    Returns:
        EnrollResponse with success status and details
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
        # Get enrollment service
        enroll_service = get_enrollment_service()
        
        # Extract frames from video
        frames, total_frames = enroll_service.enroll_from_video(
            video_path=tmp_path,
            user_id="",  # Will be generated
            name=name,
            code=code
        )
        
        if len(frames) == 0:
            return EnrollResponse(
                success=False,
                user_id=None,
                message="No frames could be extracted from video",
                embeddings_count=0,
                frames_processed=0,
                valid_frames=0
            )
        
        # Process enrollment
        result = await enroll_service.enroll_from_frames(
            frames=frames,
            user_id="",  # Will be auto-generated
            name=name,
            code=code,
            session=session
        )
        
        return EnrollResponse(
            success=result.success,
            user_id=result.user_id,
            message=result.message,
            embeddings_count=result.embeddings_count,
            frames_processed=result.frames_processed,
            valid_frames=result.valid_frames
        )
        
    finally:
        # Clean up temp file
        os.unlink(tmp_path)


@router.post("/enroll/frames", response_model=EnrollResponse)
async def enroll_from_frames(
    name: str = Form(...),
    code: str = Form(...),
    images: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_async_session)
):
    """
    Enroll a new user with multiple image frames.
    
    Alternative to video upload - send 10+ face images directly.
    
    Args:
        name: User's full name
        code: Employee/Student code (unique)
        images: List of face images
    
    Returns:
        EnrollResponse with success status
    """
    if len(images) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 5 images are required"
        )
    
    # Decode images
    frames = []
    for img_file in images:
        content = await img_file.read()
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frames.append(frame)
    
    if len(frames) < 5:
        return EnrollResponse(
            success=False,
            user_id=None,
            message="Not enough valid images",
            embeddings_count=0,
            frames_processed=len(images),
            valid_frames=len(frames)
        )
    
    # Process enrollment
    enroll_service = get_enrollment_service()
    result = await enroll_service.enroll_from_frames(
        frames=frames,
        user_id="",
        name=name,
        code=code,
        session=session
    )
    
    return EnrollResponse(
        success=result.success,
        user_id=result.user_id,
        message=result.message,
        embeddings_count=result.embeddings_count,
        frames_processed=result.frames_processed,
        valid_frames=result.valid_frames
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    session: AsyncSession = Depends(get_async_session)
):
    """
    List all enrolled users.
    
    Returns:
        UserListResponse with list of users
    """
    users = await crud.get_all_users(session)
    
    user_responses = []
    for user in users:
        templates = await crud.get_user_templates(session, user.user_id)
        user_responses.append(UserResponse(
            user_id=user.user_id,
            name=user.name,
            code=user.code,
            created_at=user.created_at,
            embeddings_count=len(templates)
        ))
    
    return UserListResponse(
        users=user_responses,
        total=len(user_responses)
    )


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Get user by ID.
    
    Args:
        user_id: User's unique ID
    
    Returns:
        UserResponse with user details
    """
    user = await crud.get_user_by_id(session, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    templates = await crud.get_user_templates(session, user.user_id)
    
    return UserResponse(
        user_id=user.user_id,
        name=user.name,
        code=user.code,
        created_at=user.created_at,
        embeddings_count=len(templates)
    )


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    session: AsyncSession = Depends(get_async_session)
):
    """
    Delete a user and their face templates.
    
    Args:
        user_id: User's unique ID
    
    Returns:
        Success message
    """
    user = await crud.get_user_by_id(session, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Remove from recognition module
    face_recognition = get_face_recognition()
    face_recognition.unregister_user(user_id)
    
    # Delete from database
    await crud.delete_user(session, user_id)
    
    return {"message": f"User {user.name} ({user.code}) deleted successfully"}
