"""
video_io.py

Small wrappers around OpenCV's VideoCapture and VideoWriter to provide a
consistent, typed interface used by the pipeline. The wrappers simplify error
handling and make tests/easy swaps possible.
"""

from typing import Optional, Tuple, Union
import cv2


class VideoReader:
    """Simple video reader wrapper.

    - Accepts an integer camera index or a path to a video file.
    - `read()` returns a BGR frame or None when the stream ends.
    - `get_size()` and `get_fps()` provide convenient accessors.
    """

    def __init__(self, source: Union[int, str] = 0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def get_size(self) -> Tuple[int, int]:
        """Return (width, height) of the capture or (0,0) if unknown."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return w, h

    def get_fps(self) -> float:
        """Return capture FPS, with a sensible default fallback."""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def release(self):
        if self.cap:
            self.cap.release()


class VideoWriter:
    """Simple video writer wrapper using mp4v fourcc.

    - If `path` is None, no file is written and methods are no-ops.
    """

    def __init__(self, path: Optional[str], frame_size: Tuple[int, int], fps: float):
        self.writer = None
        if path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(path, fourcc, fps, frame_size)

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
