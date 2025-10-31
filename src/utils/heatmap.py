import cv2
import numpy as np
from .paths import ensure_dir, ensure_odd_kernel


def clamp_array_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clamp/normalize a float array to uint8 0..255.

    Replaces NaN/Inf with zeros, scales by max value if max>0, and
    returns a 2D uint8 array suitable for colormap application.
    """
    a = np.nan_to_num(arr, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(a.max()) if a.size else 0.0
    if maxv <= 0:
        return (a * 0).astype(np.uint8)
    norm = (a / maxv * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def heatmap_colorize(acc: np.ndarray, kernel: int = 31) -> np.ndarray:
    """Convert a 2D accumulator (float) into a BGR heatmap uint8 image.

    - Applies Gaussian blur with the given odd kernel (kernel >=1).
    - Normalizes the result to 0..255 and applies COLORMAP_JET.
    Returns an HxWx3 uint8 BGR image.
    """
    if acc is None:
        raise ValueError("accumulator is None")
    if not isinstance(acc, np.ndarray):
        acc = np.asarray(acc, dtype=np.float32)
    else:
        acc = acc.astype(np.float32, copy=True)

    k = ensure_odd_kernel(kernel, default=1)
    vis = acc
    if k > 1:
        try:
            vis = cv2.GaussianBlur(vis, (k, k), 0)
        except Exception:
            # if blur fails, continue with unblurred
            vis = acc

    norm = clamp_array_to_uint8(vis)
    heat_col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return heat_col


def save_heatmap_png(acc: np.ndarray, path: str, kernel: int = 31) -> None:
    """Colorize accumulator and save as PNG to `path`. Creates parent dir."""
    if not path:
        raise ValueError("path must be provided")
    ensure_dir(path)
    img = heatmap_colorize(acc, kernel=kernel)
    # use cv2.imwrite (BGR)
    cv2.imwrite(path, img)
