#!/bin/bash
# Run the Face Recognition Attendance System

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
