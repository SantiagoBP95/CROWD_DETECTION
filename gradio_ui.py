"""Lightweight Gradio launcher for CROWD_DETECTION.

This file provides a clean, single-entry Gradio app. Run with:

    python gradio_ui.py
"""
import os
import sys
import time
from typing import List, Dict

import gradio as gr

# Ensure project root on sys.path
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.cli import parse_args as parse_yaml_args
from src.utils import process_and_write_frame, open_video_io, cleanup_resources, normalize_path, ensure_dir
from src.detector import Detector, DetectorConfig
from src.overlay import Overlay
from src.metrics import RunningCount
from src.utils import FPSSmoother
import numpy as np


def _make_video_component(kind: str, label: str):
    # Try robust signatures
    try:
        return gr.Video(sources=[kind], label=label)
    except TypeError:
        try:
            return gr.Video(source=kind, label=label)
        except TypeError:
            return gr.File(label=label)


def _make_args_for_run(out_dir: str, basename: str, selected_outputs: List[str], extra: Dict):
    args = parse_yaml_args()
    ensure_dir(out_dir)
    args.out = os.path.join(out_dir, f"{basename}.mp4")
    args.save = any(x in selected_outputs for x in ("Detections video", "Heatmap video"))
    args.csv = os.path.join(out_dir, f"{basename}.csv") if "CSV" in selected_outputs else None
    args.heatmap = "Heatmap video" in selected_outputs or "Heatmap png" in selected_outputs
    if extra.get("heatmap_kernel") is not None:
        args.heatmap_kernel = int(extra.get("heatmap_kernel"))
    if extra.get("heatmap_decay") is not None:
        args.heatmap_decay = float(extra.get("heatmap_decay"))
    return args


def process_video_file(source_path: str, selected_outputs: List[str], extra: Dict):
    ts = int(time.time())
    basename = f"gradio_{ts}"
    out_dir = os.path.join("output", "gradio", str(ts))
    os.makedirs(out_dir, exist_ok=True)

    args = _make_args_for_run(out_dir, basename, selected_outputs, extra)

    device = args.device if hasattr(args, 'device') else None
    cfg = DetectorConfig(model_path=normalize_path(args.model, ROOT), img_size=args.imgsz, conf_thres=args.conf, iou_thres=args.iou, device=device, class_whitelist=None, max_det=args.max_det, agnostic_nms=args.agnostic_nms)
    detector = Detector(cfg)
    overlay = Overlay()
    fps_sm = FPSSmoother(window_sec=0.5)
    people_counter = RunningCount()

    from src.video_io import VideoReader

    try:
        reader = VideoReader(source_path)
    except Exception as e:
        return {"error": str(e)}

    w, h = reader.get_size()
    fps = reader.get_fps()
    if w == 0 or h == 0:
        frame = reader.read()
        if frame is None:
            reader.release()
            return {"error": "no frames"}
        h, w = frame.shape[:2]

    reader, writer, heatmap_video_writer, summary_acc, w, h, fps, heatmap_video_path, heatmap_frames_written, heatmap_warned = open_video_io(source_path, args.out, ROOT, args.save, reader=reader, w=w, h=h, fps=fps)
    summary_acc_raw = np.zeros_like(summary_acc)
    metrics = []
    frame_idx = 0

    try:
        while True:
            f = reader.read()
            if f is None:
                break
            heatmap_frames_written, heatmap_warned = process_and_write_frame(
                f,
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
            # collect per-frame metrics (so CSV can be written later)
            try:
                sum_acc = float(summary_acc_raw.sum()) if summary_acc_raw is not None else 0.0
            except Exception:
                sum_acc = 0.0
            try:
                max_acc = float(summary_acc_raw.max()) if summary_acc_raw is not None else 0.0
            except Exception:
                max_acc = 0.0
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
            frame_idx += 1
    finally:
        # Ensure we store a PNG derived from the raw accumulator (without decay)
        # in case the pipeline or cleanup uses the decayed accumulator.
        try:
            from src.utils import save_heatmap_png
            hpng_raw = os.path.join(os.path.dirname(args.out), f"heatmap_raw_{os.path.splitext(os.path.basename(args.out))[0]}.png")
            save_heatmap_png(summary_acc_raw, hpng_raw, kernel=args.heatmap_kernel)
        except Exception:
            hpng_raw = None

        cleanup_resources(
            reader,
            writer,
            heatmap_video_writer,
            heatmap_video_path,
            heatmap_frames_written,
            heatmap_warned,
            summary_acc=summary_acc,
            summary_acc_raw=summary_acc_raw,
            out_path=args.out,
            kernel=args.heatmap_kernel,
            metrics=metrics,
            csv_path=args.csv,
        )

    results = {}
    if os.path.exists(args.out):
        results['detections'] = args.out
    # Prefer the raw accumulator PNG if we created it, otherwise fallback to the one
    # created by cleanup_resources.
    hpng_raw_path = os.path.join(os.path.dirname(args.out), f"heatmap_raw_{os.path.splitext(os.path.basename(args.out))[0]}.png")
    if os.path.exists(hpng_raw_path):
        results['heatmap_png'] = hpng_raw_path
    else:
        hpng = os.path.join(os.path.dirname(args.out), f"heatmap_{os.path.splitext(os.path.basename(args.out))[0]}.png")
        if os.path.exists(hpng):
            results['heatmap_png'] = hpng
    if heatmap_video_path and os.path.exists(heatmap_video_path):
        results['heatmap_video'] = heatmap_video_path
    if args.csv and os.path.exists(args.csv):
        results['csv'] = args.csv

    return results


def launch():
    with gr.Blocks() as demo:
        gr.Markdown("## Crowd Detection — UI")
        with gr.Tabs():
            with gr.TabItem("Upload"):
                upload = gr.File(file_count="single", label="Upload video file")
                outputs = gr.CheckboxGroup(choices=["Heatmap video", "Heatmap png", "CSV", "Detections video"], label="Outputs")
                kernel = gr.Number(value=31, label="Heatmap kernel")
                decay = gr.Number(value=0.9, label="Heatmap decay")
                run = gr.Button("Run")
                status = gr.Textbox(label="Status", lines=6)
                # separate outputs
                detections_play = gr.Video(label="Detections (playable)")
                detections_file = gr.File(label="Detections (download)")
                heatmap_play = gr.Video(label="Heatmap video (playable)")
                heatmap_play_file = gr.File(label="Heatmap video (download)")
                heatmap_png_preview = gr.Image(label="Heatmap PNG preview")
                heatmap_png_file = gr.File(label="Heatmap PNG (download)")
                csv_file = gr.File(label="CSV (download)")

                def run_upload(file, outs, k, d):
                    if not file:
                        return "No file", None, None, None, None, None, None
                    res = process_video_file(file.name if hasattr(file, 'name') else file, outs or [], {"heatmap_kernel": k, "heatmap_decay": d})
                    if res.get('error'):
                        return res['error'], None, None, None, None, None, None
                    # collect paths
                    det = res.get('detections')
                    hvid = res.get('heatmap_video')
                    hpng = res.get('heatmap_png')
                    csvp = res.get('csv')
                    # prepare preview image for heatmap png
                    img = None
                    if hpng and os.path.exists(hpng):
                        try:
                            from PIL import Image
                            img = Image.open(hpng)
                        except Exception:
                            img = None

                    return (
                        "Done",
                        det if det and os.path.exists(det) else None,
                        det if det and os.path.exists(det) else None,
                        hvid if hvid and os.path.exists(hvid) else None,
                        hvid if hvid and os.path.exists(hvid) else None,
                        img,
                        hpng if hpng and os.path.exists(hpng) else None,
                        csvp if csvp and os.path.exists(csvp) else None,
                    )

                run.click(
                    run_upload,
                    inputs=[upload, outputs, kernel, decay],
                    outputs=[status, detections_play, detections_file, heatmap_play, heatmap_play_file, heatmap_png_preview, heatmap_png_file, csv_file],
                )

            with gr.TabItem("Camera"):
                cam = _make_video_component('webcam', 'Record from webcam')
                outputs2 = gr.CheckboxGroup(choices=["Heatmap video", "Heatmap png", "CSV", "Detections video"], label="Outputs")
                kernel2 = gr.Number(value=31, label="Heatmap kernel")
                decay2 = gr.Number(value=0.9, label="Heatmap decay")
                run2 = gr.Button("Run")
                status2 = gr.Textbox(label="Status", lines=6)
                detections_play2 = gr.Video(label="Detections (playable)")
                detections_file2 = gr.File(label="Detections (download)")
                heatmap_play2 = gr.Video(label="Heatmap video (playable)")
                heatmap_play_file2 = gr.File(label="Heatmap video (download)")
                heatmap_png_preview2 = gr.Image(label="Heatmap PNG preview")
                heatmap_png_file2 = gr.File(label="Heatmap PNG (download)")
                csv_file2 = gr.File(label="CSV (download)")

                def run_cam(video_path, outs, k, d):
                    if not video_path:
                        return "No camera clip", None, None, None, None, None, None
                    res = process_video_file(video_path, outs or [], {"heatmap_kernel": k, "heatmap_decay": d})
                    if res.get('error'):
                        return res['error'], None, None, None, None, None, None
                    det = res.get('detections')
                    hvid = res.get('heatmap_video')
                    hpng = res.get('heatmap_png')
                    csvp = res.get('csv')
                    img = None
                    if hpng and os.path.exists(hpng):
                        try:
                            from PIL import Image
                            img = Image.open(hpng)
                        except Exception:
                            img = None
                    return (
                        "Done",
                        det if det and os.path.exists(det) else None,
                        det if det and os.path.exists(det) else None,
                        hvid if hvid and os.path.exists(hvid) else None,
                        hvid if hvid and os.path.exists(hvid) else None,
                        img,
                        hpng if hpng and os.path.exists(hpng) else None,
                        csvp if csvp and os.path.exists(csvp) else None,
                    )

                run2.click(
                    run_cam,
                    inputs=[cam, outputs2, kernel2, decay2],
                    outputs=[status2, detections_play2, detections_file2, heatmap_play2, heatmap_play_file2, heatmap_png_preview2, heatmap_png_file2, csv_file2],
                )

    demo.launch()


if __name__ == "__main__":
    launch()
