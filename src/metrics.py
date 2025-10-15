from dataclasses import dataclass

@dataclass
class RunningCount:
    """Mantiene el conteo actual, promedio y máximo desde que inició."""
    total: int = 0
    frames: int = 0
    max_count: int = 0
    current: int = 0

    def update(self, count: int) -> None:
        self.current = int(count)
        self.total += int(count)
        self.frames += 1
        if count > self.max_count:
            self.max_count = int(count)

    @property
    def average(self) -> float:
        if self.frames == 0:
            return 0.0
        return self.total / float(self.frames)
