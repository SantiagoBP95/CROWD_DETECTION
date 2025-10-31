import os
from typing import Optional, List


def normalize_path(path: Optional[str], base_dir: str) -> Optional[str]:
    """Normalize a path relative to base_dir.

    - If ``path`` is falsy (None or empty) returns None.
    - Expands user (~) and converts relative paths to absolute using ``base_dir``.
    """
    if not path:
        return None
    p = os.path.expanduser(path)
    if not os.path.isabs(p):
        p = os.path.normpath(os.path.join(base_dir, p))
    return os.path.abspath(p)


def ensure_dir(path: Optional[str]) -> None:
    """Ensure the directory for ``path`` exists.

    If ``path`` is a file path, creates its parent directory. If it's a
    directory path, creates it. Does nothing when ``path`` is falsy.
    """
    if not path:
        return
    p = os.path.expanduser(path)
    p = os.path.normpath(p)
    # If the path already exists and is a directory, nothing to do.
    if os.path.isdir(p):
        return

    # Heurística: si la ruta tiene extensión (ej. .mp4, .png) la tratamos como archivo
    # y creamos su directorio padre; si no tiene extensión asumimos que es un directorio.
    base, ext = os.path.splitext(p)
    if ext:
        parent = os.path.dirname(p)
    else:
        parent = p

    if parent:
        os.makedirs(parent, exist_ok=True)


def parse_classes(classes_str: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated class list into a list of stripped names.

    Returns None when input is falsy.
    """
    if not classes_str:
        return None
    parts = [s.strip() for s in classes_str.split(",")]
    parts = [p for p in parts if p]
    return parts if parts else None


def ensure_odd_kernel(k: Optional[int], default: int = 1) -> int:
    """Return a valid odd kernel size >= 1 for Gaussian blur.

    Accepts None or non-positive values and returns ``default`` coerced to
    a valid odd integer when needed.
    """
    if k is None:
        k = default
    try:
        ki = int(k)
    except Exception:
        ki = int(default)
    if ki < 1:
        ki = 1
    if ki % 2 == 0:
        ki += 1
    return ki
