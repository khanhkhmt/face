"""
Face Recognition Attendance System - Main API Application.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routers import enroll, attendance, config as config_router
from api.schemas import HealthResponse
from database.models import create_tables_sync
from database.session import get_sync_session
from database import crud
from config import config
from modules.face_recognition import get_face_recognition


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Face Recognition Attendance System...")
    
    # Create database tables
    logger.info("Creating database tables...")
    create_tables_sync(config.DATABASE_URL)
    
    # Create snapshot directory
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    
    # Load existing users into face recognition module
    logger.info("Loading enrolled users...")
    session = get_sync_session()
    try:
        users_with_templates = crud.get_all_templates_with_users_sync(session)
        get_face_recognition().load_users_from_db(users_with_templates)
        logger.info(f"Loaded {get_face_recognition().get_user_count()} users")
    finally:
        session.close()
    
    logger.info("System ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Face Recognition Attendance System",
    description="""
    Hệ thống điểm danh bằng nhận diện khuôn mặt với chống giả mạo.
    
    ## Features
    - Face enrollment with anti-spoofing verification
    - Attendance checking with SERIAL and PARALLEL modes
    - Temporal smoothing for robust anti-spoofing
    - Duplicate attendance detection
    - Real-time WebSocket attendance
    
    ## Pipeline Modes
    - **SERIAL**: Detection → Anti-Spoofing → Recognition
    - **PARALLEL**: Detection → [Anti-Spoofing || Recognition] → Score Fusion
    """,
    version="1.0.0",
    lifespan=lifespan
)


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(enroll.router)
app.include_router(attendance.router)
app.include_router(config_router.router)


# Static files
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Health check
@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and basic statistics.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        mode=config.MODE,
        registered_users=get_face_recognition().get_user_count(),
        database_connected=True
    )


# Root endpoint - serve frontend
@app.get("/", tags=["frontend"])
async def root():
    """Serve the main frontend page."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Face Recognition Attendance System API", "docs": "/docs"}


# API info
@app.get("/api", tags=["system"])
async def api_info():
    """Get API information."""
    return {
        "name": "Face Recognition Attendance System",
        "version": "1.0.0",
        "mode": config.MODE,
        "endpoints": {
            "enroll": "/api/enroll",
            "attendance": "/api/attendance",
            "users": "/api/users",
            "logs": "/api/attendance/logs",
            "config": "/api/config",
            "websocket": "/ws/attendance"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
