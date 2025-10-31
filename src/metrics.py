from dataclasses import dataclass


@dataclass
class RunningCount:
    """Simple running counter that tracks total, frames, maximum and current.

    Usage:
    - call `update(count)` each frame to record the per-frame count.
    - `average` property returns the average count per frame.
    """
    total: int = 0
    frames: int = 0
    max_count: int = 0
    current: int = 0

    def update(self, count: int) -> None:
        """Update internal state with the latest per-frame `count`."""
        self.current = int(count)
        self.total += int(count)
        self.frames += 1
        if count > self.max_count:
            self.max_count = int(count)

    @property
    def average(self) -> float:
        """Return average count per frame (0.0 if no frames recorded)."""
        if self.frames == 0:
            return 0.0
        return self.total / float(self.frames)
