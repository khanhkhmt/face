"""
Real-time camera capture module.
"""
import cv2
import numpy as np
from typing import Generator, Optional, Tuple
from threading import Thread
from queue import Queue
import time


class CameraCapture:
    """
    Camera capture with frame buffering.
    Supports async frame retrieval for real-time processing.
    """
    
    def __init__(
        self,
        camera_id: int = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        buffer_size: int = 10
    ):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            buffer_size: Frame buffer size
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.buffer_size = buffer_size
        
        self._cap = None
        self._frame_queue = Queue(maxsize=buffer_size)
        self._running = False
        self._thread = None
    
    def start(self) -> bool:
        """Start camera capture."""
        self._cap = cv2.VideoCapture(self.camera_id)
        
        if not self._cap.isOpened():
            return False
        
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        self._running = True
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop(self):
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()
    
    def _capture_loop(self):
        """Internal capture loop running in separate thread."""
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                # Remove old frame if queue is full
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except:
                        pass
                
                try:
                    self._frame_queue.put_nowait(frame)
                except:
                    pass
            
            time.sleep(1.0 / self.fps)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame from buffer."""
        try:
            return self._frame_queue.get_nowait()
        except:
            return None
    
    def get_frames(self, count: int, interval_ms: int = 100) -> list:
        """
        Get multiple frames with specified interval.
        
        Args:
            count: Number of frames to collect
            interval_ms: Interval between frames in milliseconds
            
        Returns:
            List of frames
        """
        frames = []
        for _ in range(count):
            frame = self.get_frame()
            if frame is not None:
                frames.append(frame)
            time.sleep(interval_ms / 1000.0)
        return frames
    
    def stream(self) -> Generator[np.ndarray, None, None]:
        """Generate frames continuously."""
        while self._running:
            frame = self.get_frame()
            if frame is not None:
                yield frame
    
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._running
    
    @property
    def frame_count(self) -> int:
        """Number of frames in buffer."""
        return self._frame_queue.qsize()


def get_available_cameras() -> list:
    """Get list of available camera IDs."""
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available
