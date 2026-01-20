# Face Recognition Attendance System with Anti-Spoofing

Hệ thống điểm danh bằng nhận diện khuôn mặt với chống giả mạo.

## Features

- ✅ Face enrollment with anti-spoofing verification
- ✅ Attendance checking with SERIAL and PARALLEL modes
- ✅ Temporal smoothing for robust anti-spoofing
- ✅ Duplicate attendance detection
- ✅ Real-time WebSocket attendance
- ✅ Beautiful web UI

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Run the Server

```bash
# Using run script
chmod +x run.sh
./run.sh

# Or directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open Web UI

Navigate to: http://localhost:8000

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/enroll` | Enroll user with video |
| POST | `/api/enroll/frames` | Enroll with image frames |
| POST | `/api/attendance` | Check attendance with video |
| POST | `/api/attendance/image` | Check with single image |
| POST | `/api/attendance/frames` | Check with multiple frames |
| GET | `/api/users` | List enrolled users |
| GET | `/api/attendance/logs` | Get attendance logs |
| GET | `/api/config` | Get configuration |
| PUT | `/api/config` | Update configuration |
| WS | `/ws/attendance` | Real-time attendance |

## Pipeline Modes

### SERIAL Mode (Default)
```
Camera → Face Detection → Anti-Spoofing → Face Recognition → Decision
```

### PARALLEL Mode
```
Camera → Face Detection → [Anti-Spoofing || Face Recognition] → Score Fusion → Decision
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODE` | SERIAL | Pipeline mode |
| `T_FAS` | 0.5 | Anti-spoofing threshold |
| `T_FR` | 0.4 | Face recognition threshold |
| `TEMPORAL_FRAMES` | 10 | Frames for temporal smoothing |
| `DUPLICATE_WINDOW` | 5 min | Duplicate check window |

## Project Structure

```
prj_face/
├── api/                    # FastAPI application
│   ├── main.py            # Main app entry
│   ├── schemas.py         # Pydantic models
│   └── routers/           # API routes
│       ├── enroll.py
│       ├── attendance.py
│       └── config.py
├── modules/               # Face processing modules
│   ├── face_detector.py   # Face detection
│   ├── anti_spoofing.py   # Anti-spoofing
│   ├── face_recognition.py # Face recognition
│   └── score_fusion.py    # Score fusion
├── services/              # Business logic
│   ├── enroll_service.py
│   └── attendance_service.py
├── database/              # Database layer
│   ├── models.py
│   ├── crud.py
│   └── session.py
├── static/                # Frontend
│   └── index.html
├── config.py              # Configuration
├── requirements.txt
├── run.sh
└── README.md
```

## License

MIT License
