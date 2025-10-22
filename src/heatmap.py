from typing import Deque, Tuple, List
import collections
import numpy as np
import cv2


class HeatmapAccumulator:
    """Acumula centros de detección y genera un mapa de calor superponible.

    Uso:
      hm = HeatmapAccumulator(frame_size=(w,h), decay=0.9, kernel=31)
      hm.add_points([(x1,y1),(x2,y2)])
      out = hm.overlay_heatmap(frame, alpha=0.6, colormap=cv2.COLORMAP_JET)

    Parámetros:
      - frame_size: (width, height) del frame donde se superpondrá el heatmap
      - decay: factor de atenuación por frame (0..1). Valores cercanos a 1 mantienen historial.
      - kernel: tamaño de kernel para gaussian blur del acumulador (impar)
    """

    def __init__(self, frame_size: Tuple[int, int], decay: float = 0.9, kernel: int = 31, max_history: int = 1000):
        w, h = frame_size
        # mapa acumulador en float32
        self.w = int(w)
        self.h = int(h)
        self.acc = np.zeros((self.h, self.w), dtype=np.float32)
        # asegurar decay en [0,1]
        self.decay = float(decay)
        if self.decay < 0.0:
            self.decay = 0.0
        elif self.decay > 1.0:
            self.decay = 1.0
        # kernel debe ser entero impar y al menos 1
        k = int(kernel)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1
        self.kernel = k
        self.history: Deque[Tuple[int, int]] = collections.deque(maxlen=max_history)

    def add_points(self, points: List[Tuple[int, int]]):
        """Añade centros (x,y) al acumulador (en coordenadas de píxeles).

        Si un punto queda fuera del frame se ignora.
        """
        for x, y in points:
            ix = int(round(x))
            iy = int(round(y))
            if 0 <= ix < self.w and 0 <= iy < self.h:
                # incrementar con un valor unitario
                self.acc[iy, ix] += 1.0
                self.history.append((ix, iy))

    def decay_step(self):
        """Aplicar decaimiento exponencial al acumulador."""
        self.acc *= self.decay

    def get_heatmap(self) -> np.ndarray:
        """Retorna el heatmap normalizado en uint8 (0..255)."""
        # aplicar blur para suavizar
        acc_copy = self.acc.copy()
        if self.kernel > 1:
            acc_copy = cv2.GaussianBlur(acc_copy, (self.kernel, self.kernel), 0)
        # normalizar a 0..255
        maxv = acc_copy.max()
        if maxv <= 0:
            return np.zeros((self.h, self.w), dtype=np.uint8)
        nm = (acc_copy / maxv * 255.0).clip(0, 255).astype(np.uint8)
        return nm

    def overlay_heatmap(self, frame: np.ndarray, alpha: float = 0.6, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Superpone el heatmap sobre el frame (BGR) y retorna la imagen compuesta."""
        heat = self.get_heatmap()
        if heat.sum() == 0:
            # devolver copia para evitar mutaciones inesperadas
            return frame.copy()
        # asegurar alpha válido
        try:
            alpha = float(alpha)
        except Exception:
            alpha = 0.6
        alpha = max(0.0, min(1.0, alpha))
        colored = cv2.applyColorMap(heat, colormap)
        # redimensionar si es necesario
        if colored.shape[:2] != frame.shape[:2]:
            colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]))

        # asegurar que el frame es BGR uint8
        f = frame
        if f.dtype != np.uint8:
            # copiar y convertir
            f = (f.astype(np.float32)).clip(0, 255).astype(np.uint8)
        if f.ndim == 2:
            # convertir grayscale a BGR
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        elif f.shape[2] == 1:
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

        # composición
        out = cv2.addWeighted(colored, alpha, f, 1.0 - alpha, 0)
        return out

    def step(self, points: List[Tuple[int, int]] = None):
        """Conveniencia: añadir puntos y decaer (llamar cada frame)."""
        # aplicar decaimiento de historial previo, luego añadir los puntos del frame actual
        self.decay_step()
        if points:
            self.add_points(points)
