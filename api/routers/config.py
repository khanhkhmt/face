"""
Configuration API Router.
Allows viewing and updating system configuration.
"""
from fastapi import APIRouter, HTTPException

from api.schemas import ConfigResponse, ConfigUpdate
from config import config, PipelineMode, FusionMethod
from services.attendance_service import get_attendance_service
from modules.score_fusion import get_score_fusion


router = APIRouter(prefix="/api", tags=["configuration"])


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """
    Get current system configuration.
    
    Returns:
        ConfigResponse with all configuration values
    """
    return ConfigResponse(
        mode=config.MODE,
        fusion_method=config.FUSION_METHOD,
        t_fas=config.T_FAS,
        t_fr=config.T_FR,
        t_final=config.T_FINAL,
        t_fas_min=config.T_FAS_MIN,
        w1=config.W1,
        w2=config.W2,
        temporal_frames=config.TEMPORAL_FRAMES,
        duplicate_window_minutes=config.DUPLICATE_WINDOW_MINUTES
    )


@router.put("/config", response_model=ConfigResponse)
async def update_config(updates: ConfigUpdate):
    """
    Update system configuration.
    
    Only provided fields will be updated.
    Changes take effect immediately.
    
    Args:
        updates: Configuration updates
    
    Returns:
        Updated ConfigResponse
    """
    # Update config object
    if updates.mode is not None:
        config.MODE = updates.mode
        get_attendance_service().set_mode(PipelineMode(updates.mode))
    
    if updates.fusion_method is not None:
        config.FUSION_METHOD = updates.fusion_method
        get_score_fusion().method = FusionMethod(updates.fusion_method)
    
    if updates.t_fas is not None:
        config.T_FAS = updates.t_fas
        get_score_fusion().t_fas = updates.t_fas
    
    if updates.t_fr is not None:
        config.T_FR = updates.t_fr
        get_score_fusion().t_fr = updates.t_fr
    
    if updates.t_final is not None:
        config.T_FINAL = updates.t_final
        get_score_fusion().t_final = updates.t_final
    
    if updates.t_fas_min is not None:
        config.T_FAS_MIN = updates.t_fas_min
        get_score_fusion().t_fas_min = updates.t_fas_min
    
    if updates.w1 is not None:
        config.W1 = updates.w1
        get_score_fusion().w1 = updates.w1
    
    if updates.w2 is not None:
        config.W2 = updates.w2
        get_score_fusion().w2 = updates.w2
    
    if updates.temporal_frames is not None:
        config.TEMPORAL_FRAMES = updates.temporal_frames
    
    if updates.duplicate_window_minutes is not None:
        config.DUPLICATE_WINDOW_MINUTES = updates.duplicate_window_minutes
        get_attendance_service().duplicate_window_minutes = updates.duplicate_window_minutes
    
    return await get_config()


@router.post("/config/reset")
async def reset_config():
    """
    Reset configuration to defaults.
    
    Returns:
        Default ConfigResponse
    """
    # Reset to defaults
    config.MODE = "SERIAL"
    config.FUSION_METHOD = "AND_GATE"
    config.T_FAS = 0.5
    config.T_FR = 0.4
    config.T_FINAL = 0.6
    config.T_FAS_MIN = 0.3
    config.W1 = 0.6
    config.W2 = 0.4
    config.TEMPORAL_FRAMES = 10
    config.DUPLICATE_WINDOW_MINUTES = 5
    
    # Update services
    get_attendance_service().set_mode(PipelineMode.SERIAL)
    get_score_fusion().update_config(
        method=FusionMethod.AND_GATE,
        t_fas=0.5,
        t_fr=0.4,
        t_final=0.6,
        t_fas_min=0.3,
        w1=0.6,
        w2=0.4
    )
    
    return await get_config()
