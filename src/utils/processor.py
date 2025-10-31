import cv2
import numpy as np
from typing import Tuple
from .paths import ensure_odd_kernel
from .heatmap import heatmap_colorize


def process_and_write_frame(
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
    summary_acc_raw=None,
) -> Tuple[int, bool]:
    """Process a single frame: run inference, update accumulators and draw overlays.

    Steps:
    - Run detector.infer(frame)
    - Update people counter and accumulators
    - Draw boxes, HUD and FPS
    - Optionally write heatmap frames and annotated frames to the writers

    Returns:
        (heatmap_frames_written, heatmap_warned)
    """

    names = getattr(detector, "names", getattr(getattr(detector, "model", {}), "names", {}))
    dets = detector.infer(frame)
    current_count = len(dets)
    people_counter.update(current_count)

    centers = []
    for (x1, y1, x2, y2), conf, cid in dets:
        name = names[cid] if cid in names else f"id{cid}"
        label = f"{name} {conf:.2f}"
        overlay.draw_box(frame, (x1, y1, x2, y2), label)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        centers.append((cx, cy))

    # accumulator decay
    try:
        decay = float(args.heatmap_decay)
    except Exception:
        decay = 1.0
    if decay < 1.0:
        summary_acc *= decay

    for cx, cy in centers:
        ix = int(round(cx))
        iy = int(round(cy))
        if 0 <= ix < w and 0 <= iy < h:
            # increment both the decayable accumulator and the raw accumulator
            summary_acc[iy, ix] += 1.0
            if summary_acc_raw is not None:
                summary_acc_raw[iy, ix] += 1.0

    # heatmap frame generation
    if heatmap_video_writer is not None:
        k_vid = args.heatmap_video_kernel if args.heatmap_video_kernel is not None else args.heatmap_kernel
        k = ensure_odd_kernel(k_vid, default=1)
        try:
            heat_col = heatmap_colorize(summary_acc, kernel=k)
        except Exception:
            summary_vis = summary_acc.copy()
            if k > 1:
                try:
                    summary_vis = cv2.GaussianBlur(summary_vis, (k, k), 0)
                except Exception:
                    pass
            maxv = summary_vis.max()
            if maxv <= 0:
                norm = (summary_vis * 0).astype(np.uint8)
            else:
                norm = (summary_vis / maxv * 255.0).clip(0, 255).astype(np.uint8)
            heat_col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
            if heat_col.shape[:2] != (h, w):
                heat_col = cv2.resize(heat_col, (w, h))

        try:
            opened = False
            if hasattr(heatmap_video_writer, "writer") and heatmap_video_writer.writer is not None:
                try:
                    opened = bool(heatmap_video_writer.writer.isOpened())
                except Exception:
                    opened = True
            if opened:
                heatmap_video_writer.write(heat_col)
                heatmap_frames_written += 1
            else:
                if not heatmap_warned:
                    print(f"[WARN] heatmap_video_writer is not open; heatmap frames will not be written.")
                    heatmap_warned = True
        except Exception:
            if not heatmap_warned:
                print(f"[WARN] error while attempting to write heatmap frame")
                heatmap_warned = True

    # HUD, draw and write frame
    overlay.draw_counts(frame, people_counter.current, people_counter.average, people_counter.max_count)
    overlay.draw_fps(frame, fps_sm.tick())
    cv2.imshow("Real-time detection (YOLO)", frame)
    if writer is not None and getattr(writer, "writer", None) is not None:
        writer.write(frame)

    return heatmap_frames_written, heatmap_warned
