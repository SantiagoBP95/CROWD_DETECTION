import time


class FPSSmoother:
    """Smooths FPS over a sliding time window."""
    def __init__(self, window_sec: float = 0.5):
        self.window_sec = window_sec
        self._last = time.time()
        self._count = 0
        self.fps = 0.0

    def tick(self) -> float:
        self._count += 1
        now = time.time()
        dt = now - self._last
        if dt >= self.window_sec:
            self.fps = self._count / dt
            self._count = 0
            self._last = now
        return self.fps
