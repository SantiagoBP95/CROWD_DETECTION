"""
cli.py

Utility to parse configuration for the application. Supports reading a YAML
file with defaults and returns an argparse.Namespace compatible object used
throughout the project.
"""

import argparse
import os
import yaml


def parse_args():
    """Load configuration from YAML (or fallbacks) and return a Namespace.

    The function looks for a `--config` argument pointing to a YAML file. If
    present, values from the YAML override the built-in fallbacks. The result
    is coerced to common types (ints, floats, bools) and returned as an
    argparse.Namespace for convenience.
    """
    # Now: only read the YAML and return a Namespace built from it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "config.yml"),
        help="Path to YAML configuration file.",
    )
    known, _ = pre.parse_known_args()

    cfg_path = known.config
    yaml_defaults = {}
    try:
        if cfg_path and os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                yaml_defaults = yaml.safe_load(f) or {}
    except Exception:
        yaml_defaults = {}

    # Provide hard-coded fallbacks for keys that might be missing in YAML
    fallbacks = {
        "model": "models/best.pt",
        "source": "data/input/input2.mp4",
        "imgsz": 896,
        "conf": 0.10,
        "iou": 0.30,
        "device": None,
        "save": True,
        "out": "output/salida_detectada.mp4",
        "classes": "head",
        "max_det": 3000,
        "agnostic_nms": False,
        "heatmap": False,
        "heatmap_alpha": 0.6,
        "heatmap_decay": 0.9,
        "heatmap_kernel": 31,
        "heatmap_video_kernel": None,
        "csv": None,
        "heatmap_png_interval": None,
    }

    # Merge fallbacks with YAML (YAML wins)
    cfg = {**fallbacks, **(yaml_defaults or {})}

    # coerce common types
    try:
        cfg["imgsz"] = int(cfg.get("imgsz", fallbacks["imgsz"]))
    except Exception:
        cfg["imgsz"] = fallbacks["imgsz"]
    try:
        cfg["conf"] = float(cfg.get("conf", fallbacks["conf"]))
    except Exception:
        cfg["conf"] = fallbacks["conf"]
    try:
        cfg["iou"] = float(cfg.get("iou", fallbacks["iou"]))
    except Exception:
        cfg["iou"] = fallbacks["iou"]
    try:
        cfg["max_det"] = int(cfg.get("max_det", fallbacks["max_det"]))
    except Exception:
        cfg["max_det"] = fallbacks["max_det"]

    # Ensure boolean coercion
    for b in ("save", "agnostic_nms", "heatmap"):
        val = cfg.get(b, fallbacks[b])
        if isinstance(val, str):
            cfg[b] = val.lower() in ("1", "true", "yes", "y", "on")
        else:
            cfg[b] = bool(val)

    # coerce interval to float or None
    try:
        interval = cfg.get("heatmap_png_interval", fallbacks.get("heatmap_png_interval"))
        if interval is None:
            cfg["heatmap_png_interval"] = None
        else:
            cfg["heatmap_png_interval"] = float(interval)
            if cfg["heatmap_png_interval"] <= 0:
                cfg["heatmap_png_interval"] = None
    except Exception:
        cfg["heatmap_png_interval"] = None

    # Return an argparse-like Namespace constructed from cfg
    return argparse.Namespace(**cfg)


def parse_source(src_str: str):
    """Parse a source string into either an int (camera index) or path.

    If the provided string can be converted to an int, the int is returned so
    callers can open the corresponding camera index. Otherwise the original
    string is returned and treated as a file path.
    """
    # If it's a number, use it as a camera index.
    try:
        return int(src_str)
    except ValueError:
        return src_str
