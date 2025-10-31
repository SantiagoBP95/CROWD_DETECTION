"""
main.py

Command-line entry point for running the YOLO-based real-time detector.

This module exposes a single `main()` function which parses configuration
from `src.cli.parse_args()` (YAML-backed), initializes the detector and I/O
components, processes frames from a camera or video file, and optionally
saves a detections video, heatmap video/PNG and a CSV with per-frame metrics.

Inputs:
- command-line / YAML configuration (model path, source, img size, thresholds,
  whether to save outputs, etc.)

Outputs:
- Writes output files to disk when requested and displays a live OpenCV
  window with detection overlays (unless running headless).
"""

import os
import cv2
import numpy as np
import torch
import time

from src.detector import Detector, DetectorConfig
from src.video_io import VideoReader
from src.overlay import Overlay
from src.utils import (
    FPSSmoother,
    normalize_path,
    ensure_dir,
    parse_classes,
    open_video_io,
    process_and_write_frame,
    cleanup_resources,
)

from src.metrics import RunningCount
from src.cli import parse_args, parse_source


def main():
    """Run the real-time detection pipeline.

    No arguments; configuration is loaded from command-line / YAML via
    `src.cli.parse_args()`.

    Side effects:
    - May open the default camera or a video file and display an OpenCV window.
    - May write output files (detections video, heatmap video, heatmap PNG, CSV).
    """
    args = parse_args()

    # Resolve device (accept numeric indexes like '0' for cuda:0)
    if args.device is not None:
        device = args.device
        try:
            di = int(device)
            device = f"cuda:{di}" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Classes whitelist (None means all classes)
    class_whitelist = parse_classes(args.classes)

    # Normalize paths relative to the project base dir
    base_dir = os.path.dirname(os.path.abspath(__file__))
    args.model = normalize_path(args.model, base_dir)

    # Normalize source path if provided as a string
    if isinstance(args.source, str) and not args.source.isdigit():
        args.source = normalize_path(args.source, base_dir)
        if not os.path.exists(args.source):
            print(f"[WARN] Video source does not exist: {args.source}")

    # Fail early if model is missing
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found at: {args.model}. Provide a valid .pt via --model.")

    # Prepare detector and helpers
    cfg = DetectorConfig(
        model_path=args.model,
        img_size=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=device,
        class_whitelist=class_whitelist,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms,
    )
    detector = Detector(cfg)
    overlay = Overlay()
    fps_sm = FPSSmoother(window_sec=0.5)
    people_counter = RunningCount()
    metrics = []  # per-frame metrics collector (for optional CSV)

    # Open video source with error handling
    source = parse_source(args.source)
    try:
        reader = VideoReader(source)
    except Exception as e:
        print(f"[ERROR] Could not open video source {source}: {e}")
        return

    w, h = reader.get_size()
    fps = reader.get_fps()

    # If the reader couldn't determine size, read a frame to get dimensions
    peek_frame = None
    if w == 0 or h == 0:
        peek_frame = reader.read()
        if peek_frame is None:
            print(f"[ERROR] No frames could be read from the source: {args.source}")
            reader.release()
            return
        h, w = peek_frame.shape[:2]

    # Prepare output paths and optional CSV
    args.out = normalize_path(args.out, base_dir)
    out_path = args.out
    ensure_dir(out_path)
    out_dir = os.path.dirname(out_path)

    csv_path = None
    if hasattr(args, "csv") and args.csv:
        args.csv = normalize_path(args.csv, base_dir)
        ensure_dir(args.csv)
        csv_path = args.csv

    # Periodic PNG interval (seconds) for live snapshots
    png_interval = getattr(args, "heatmap_png_interval", None)
    last_png_time = time.time()
    live_heatmap_path = None
    if png_interval:
        live_heatmap_path = os.path.join(out_dir, "heatmap_live.png")
        ensure_dir(live_heatmap_path)

    # Open writers and get accumulators
    reader, writer, heatmap_video_writer, summary_acc, w, h, fps, heatmap_video_path, heatmap_frames_written, heatmap_warned = \
        open_video_io(source, out_path, base_dir, args.save, reader=reader, w=w, h=h, fps=fps)

    # Raw accumulator (no decay) for final PNG
    summary_acc_raw = np.zeros_like(summary_acc)

    print(f"Press 'q' to quit. heatmap={args.heatmap}, save={args.save}, out={args.out}", flush=True)

    first_iteration = True if peek_frame is not None else False
    frame_idx = 0

    try:
        while True:
            if first_iteration:
                frame = peek_frame
                first_iteration = False
            else:
                frame = reader.read()
            if frame is None:
                break

            heatmap_frames_written, heatmap_warned = process_and_write_frame(
                frame,
                detector,
                overlay,
                fps_sm,
                people_counter,
                summary_acc,
                writer,
                heatmap_video_writer,
                args,
                w,
                h,
                heatmap_frames_written,
                heatmap_warned,
                summary_acc_raw=summary_acc_raw,
            )

            # Record derived accumulator metrics
            sum_acc = float(summary_acc_raw.sum()) if summary_acc_raw is not None else 0.0
            max_acc = float(summary_acc_raw.max()) if summary_acc_raw is not None else 0.0
            metrics.append({
                "frame": frame_idx,
                "timestamp": time.time(),
                "count": people_counter.current,
                "avg": people_counter.average,
                "max": people_counter.max_count,
                "fps": float(getattr(fps_sm, "fps", 0.0)),
                "sum_acc": sum_acc,
                "max_acc": max_acc,
                "total_people": people_counter.total,
            })

            # Periodic live PNG from the raw accumulator
            if png_interval and (time.time() - last_png_time) >= png_interval:
                try:
                    from src.utils.heatmap import save_heatmap_png

                    save_heatmap_png(summary_acc_raw if summary_acc_raw is not None else summary_acc, live_heatmap_path, kernel=args.heatmap_kernel)
                    last_png_time = time.time()
                except Exception:
                    pass

            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # Pass accumulators so cleanup can save PNG / CSV summaries
        cleanup_resources(
            reader,
            writer,
            heatmap_video_writer,
            heatmap_video_path,
            heatmap_frames_written,
            heatmap_warned,
            summary_acc=summary_acc,
            summary_acc_raw=summary_acc_raw,
            out_path=out_path,
            kernel=args.heatmap_kernel,
            metrics=metrics,
            csv_path=csv_path,
        )


if __name__ == "__main__":
    main()
