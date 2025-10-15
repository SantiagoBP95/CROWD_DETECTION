from typing import Tuple
import cv2

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX

class Overlay:
    def __init__(self, font_scale: float = 0.6, font_thickness: int = 2):
        self.font = DEFAULT_FONT
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_box(
        self, frame, box: Tuple[int, int, int, int], label: str, color=(255, 0, 0)
    ) -> None:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), self.font, self.font_scale, (255, 255, 255),
                    self.font_thickness, cv2.LINE_AA)

    def draw_fps(self, frame, fps: float) -> None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), self.font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    def draw_counts(self, frame, current: int, average: float, max_count: int) -> None:
        """Pequeño panel con Current / Average / Maximum."""
        pad = 8
        lines = [
            f"Current: {current}",
            f"Average: {average:.1f}",
            f"Maximum: {max_count}",
        ]
        # medir el panel
        widths, heights = [], []
        for text in lines:
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.6, self.font_thickness)
            widths.append(tw); heights.append(th)
        panel_w = max(widths) + pad*2
        panel_h = sum(heights) + pad*2 + (len(lines)-1)*4

        x0, y0 = 10, 50
        cv2.rectangle(frame, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        y = y0 + pad + heights[0]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (x0 + pad, y), self.font, 0.6, (255, 255, 255),
                        self.font_thickness, cv2.LINE_AA)
            if i < len(lines) - 1:
                y += heights[i+1] + 4
