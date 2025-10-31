import os
import csv
import time
from typing import Optional, List
import cv2
import numpy as np
from src.detector import Detector, DetectorConfig
from src.video_io import VideoReader, VideoWriter
from src.overlay import Overlay
from src.metrics import RunningCount

# re-export / adapters: move specialized helpers to submodules for clarity
from .paths import normalize_path, ensure_dir, parse_classes, ensure_odd_kernel
from .heatmap import heatmap_colorize, save_heatmap_png
from .fps import FPSSmoother
from .processor import process_and_write_frame


# the functions above were moved to submodules:
#  - src.utils.paths
#  - src.utils.heatmap
#  - src.utils.fps
# We keep this module as a compatibility layer and the higher-level
# pipeline functions below.


def init_components(cfg: 'DetectorConfig'):
    """Inicializa detector, overlay, fps smoother y contador de personas.

    Devuelve (detector, overlay, fps_sm, people_counter, names).
    """
    detector = Detector(cfg)
    overlay = Overlay()
    fps_sm = FPSSmoother(window_sec=0.5)
    people_counter = RunningCount()
    names = getattr(detector, 'names', getattr(detector, 'model', {}).get('names', {}))
    return detector, overlay, fps_sm, people_counter, names


def open_video_io(source, out_path, base_dir, save_flag, reader=None, w=None, h=None, fps=None):
    """Abre reader y writers, devuelve (reader, writer, heatmap_video_writer, summary_acc, w, h, fps, heatmap_video_path, heatmap_frames_written, heatmap_warned).

    ``source`` puede ser int o ruta. Si reader no determina tamaño, caller deberá leer un frame.
    """
    if reader is None:
        reader = VideoReader(source)
    # obtener tamaño y fps del reader
    r_w, r_h = reader.get_size()
    r_fps = reader.get_fps()
    # use provided w/h/fps cuando sean None
    w = r_w if (w is None or r_w == 0) else (w if w is not None else r_w)
    h = r_h if (h is None or r_h == 0) else (h if h is not None else r_h)
    fps = r_fps if (fps is None) else fps

    writer = VideoWriter(out_path if save_flag else None, (w, h), fps)

    summary_acc = np.zeros((h, w), dtype=np.float32)
    heatmap_video_writer = None
    heatmap_video_path = None
    heatmap_frames_written = 0
    heatmap_warned = False
    if save_flag:
        heatmap_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else base_dir
        heatmap_video_path = os.path.join(heatmap_dir, f"heatmap_{os.path.splitext(os.path.basename(out_path))[0]}.mp4")
        heatmap_video_writer = VideoWriter(heatmap_video_path, (w, h), fps)
        # comprobar apertura
        try:
            ok = False
            if hasattr(heatmap_video_writer, 'writer') and heatmap_video_writer.writer is not None:
                try:
                    ok = bool(heatmap_video_writer.writer.isOpened())
                except Exception:
                    ok = True
            if not ok:
                print(f"[WARN] No se pudo abrir el writer para el video de heatmap: {heatmap_video_path}")
                heatmap_video_writer = None
            else:
                print(f"[INFO] Se creará heatmap video en: {heatmap_video_path}")
        except Exception:
            pass

    return reader, writer, heatmap_video_writer, summary_acc, w, h, fps, heatmap_video_path, heatmap_frames_written, heatmap_warned


# process_and_write_frame moved to src.utils.processor


def cleanup_resources(reader, writer, heatmap_video_writer, heatmap_video_path, heatmap_frames_written, heatmap_warned, summary_acc=None, summary_acc_raw=None, out_path=None, kernel=None, metrics=None, csv_path=None):
    """Libera recursos, guarda heatmap PNG opcional, CSV de métricas y reporta resumen.

    Parameters
    - metrics: lista de diccionarios con métricas por frame (opcional)
    - csv_path: ruta donde escribir el CSV si se desea (opcional)
    """
    try:
        reader.release()
    except Exception:
        pass
    try:
        writer.release()
    except Exception:
        pass
    if heatmap_video_writer is not None:
        try:
            heatmap_video_writer.release()
            print(f"[INFO] Heatmap video guardado en: {heatmap_video_path}")
        except Exception:
            pass
        try:
            print(f"[INFO] Frames de heatmap escritos: {heatmap_frames_written}", flush=True)
            if os.path.exists(heatmap_video_path):
                size = os.path.getsize(heatmap_video_path)
                print(f"[INFO] Tamaño del archivo: {size} bytes -> {heatmap_video_path}", flush=True)
        except Exception:
            pass

    # guardar PNG resumen opcional. Preferimos el acumulador 'raw' (sin decaimiento)
    acc_to_save = None
    if summary_acc_raw is not None:
        acc_to_save = summary_acc_raw
    elif summary_acc is not None:
        acc_to_save = summary_acc

    if acc_to_save is not None and out_path is not None:
        try:
            heatmap_png_path = os.path.join(os.path.dirname(out_path), f"heatmap_{os.path.splitext(os.path.basename(out_path))[0]}.png")
            save_heatmap_png(acc_to_save, heatmap_png_path, kernel=kernel if kernel is not None else 31)
            print(f"[INFO] Heatmap PNG guardado en: {heatmap_png_path}")
        except Exception:
            pass

    # guardar CSV opcional (incluye métricas derivadas del acumulador)
    if csv_path:
        try:
            # asegurar directorio padre
            ensure_dir(csv_path)
            fieldnames = ["frame", "timestamp", "count", "avg", "max", "fps", "sum_acc", "max_acc", "total_people"]
            with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                writer_csv = csv.DictWriter(cf, fieldnames=fieldnames)
                writer_csv.writeheader()
                if metrics:
                    for row in metrics:
                        # asegurarse de tener solo las keys esperadas
                        writer_csv.writerow({k: row.get(k, "") for k in fieldnames})
            print(f"[INFO] CSV de métricas guardado en: {csv_path}")
        except Exception as e:
            print(f"[WARN] No se pudo guardar CSV de métricas en {csv_path}: {e}")

    cv2.destroyAllWindows()
