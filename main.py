import argparse
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
    args = parse_args()

    # Dispositivo por defecto. Aceptamos índices numéricos (0, 1, ...) y los mapeamos a cuda:<idx>
    if args.device is not None:
        device = args.device
        # permitir que el usuario pase '0' o 0 para indicar cuda:0
        try:
            di = int(device)
            # si hay CUDA disponible, mapear a cuda:<di>, si no, dejar como 'cpu'
            device = f"cuda:{di}" if torch.cuda.is_available() else "cpu"
        except Exception:
            # puede ser 'cpu', 'cuda:0', '0,1' u otra cadena válida aceptada por torch
            device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Lista de clases a permitir
    class_whitelist = parse_classes(args.classes)

    # Resolver rutas relativas respecto al directorio del script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Modelo: normalizar ruta relativa respecto al paquete
    args.model = normalize_path(args.model, base_dir)

    # Fuente de video: si es ruta relativa, interpretarla relativa al paquete
    if isinstance(args.source, str) and not args.source.isdigit():
        args.source = normalize_path(args.source, base_dir)
        if not os.path.exists(args.source):
            print(f"[WARN] La fuente de video no existe: {args.source}")

    # Comprobar modelo y fallar temprano con mensaje claro si falta
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"El modelo no existe en la ruta: {args.model}. Coloca el .pt allí o pasa --model con la ruta correcta.")

    # Config y objetos principales
    cfg = DetectorConfig(
        model_path=args.model,
        img_size=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=device,
        class_whitelist=class_whitelist,
        max_det=args.max_det,
        agnostic_nms=args.agnostic_nms
    )
    detector = Detector(cfg)
    overlay = Overlay()
    fps_sm = FPSSmoother(window_sec=0.5)
    people_counter = RunningCount()
    # recolector de métricas por frame (para CSV opcional)
    metrics = []

    # IO de video: abrir reader y preparar writers (usar reader existente si es necesario)
    source = parse_source(args.source)
    # Abrir la fuente de video con manejo de errores para evitar crash abrupto
    try:
        reader = VideoReader(source)
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la fuente de video {source}: {e}")
        return

    w, h = reader.get_size()
    fps = reader.get_fps()

    # If VideoReader couldn't determine size (0,0), read the first frame to get dimensions.
    peek_frame = None
    if w == 0 or h == 0:
        peek_frame = reader.read()
        if peek_frame is None:
            print(f"[ERROR] No se pudo leer ningún frame desde la fuente: {args.source}")
            reader.release()
            return
        # OpenCV frames: shape = (height, width, channels)
        h, w = peek_frame.shape[:2]

    # Resolver y asegurar la ruta de salida (--out) respecto al paquete
    args.out = normalize_path(args.out, base_dir)
    out_path = args.out
    ensure_dir(out_path)
    out_dir = os.path.dirname(out_path)

    # CSV opcional: normalizar ruta si fue proporcionada en config
    csv_path = None
    if hasattr(args, 'csv') and args.csv:
        args.csv = normalize_path(args.csv, base_dir)
        # ensure parent dir exists
        ensure_dir(args.csv)
        csv_path = args.csv

    # Intervalo (segundos) para PNG intermedio
    png_interval = getattr(args, 'heatmap_png_interval', None)
    last_png_time = time.time()
    live_heatmap_path = None
    if png_interval:
        live_heatmap_path = os.path.join(out_dir, "heatmap_live.png")
        ensure_dir(live_heatmap_path)

    # abrir writers / obtener summary_acc y estado del heatmap
    reader, writer, heatmap_video_writer, summary_acc, w, h, fps, heatmap_video_path, heatmap_frames_written, heatmap_warned = \
        open_video_io(source, out_path, base_dir, args.save, reader=reader, w=w, h=h, fps=fps)

    # acumulador sin decaimiento para guardar PNG final (si se desea)
    summary_acc_raw = np.zeros_like(summary_acc)

    # names ya obtenido desde init_components
    # Diagnostics print: show key args so user sees algo en consola
    print(f"Presiona 'q' para salir. args.heatmap={args.heatmap}, args.save={args.save}, out={args.out}", flush=True)
    # No se realiza visualización en vivo del heatmap; solo se acumula summary_acc para guardar al final
    # If we peeked one frame to determine size, process it first
    first_iteration = True if peek_frame is not None else False
    frame_idx = 0
    # Nota: no guardamos la imagen PNG resumen; solo generamos el video de heatmap

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
            # registrar métricas de este frame
            # derived accumulator metrics from raw accumulator
            sum_acc = float(summary_acc_raw.sum()) if summary_acc_raw is not None else 0.0
            max_acc = float(summary_acc_raw.max()) if summary_acc_raw is not None else 0.0
            metrics.append({
                "frame": frame_idx,
                "timestamp": time.time(),
                "count": people_counter.current,
                "avg": people_counter.average,
                "max": people_counter.max_count,
                "fps": float(getattr(fps_sm, 'fps', 0.0)),
                "sum_acc": sum_acc,
                "max_acc": max_acc,
                "total_people": people_counter.total,
            })
            # periodic live PNG (undecaied accumulator)
            if png_interval and (time.time() - last_png_time) >= png_interval:
                try:
                    from src.utils.heatmap import save_heatmap_png
                    save_heatmap_png(summary_acc_raw if summary_acc_raw is not None else summary_acc, live_heatmap_path, kernel=args.heatmap_kernel)
                    last_png_time = time.time()
                except Exception:
                    pass
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # pasar summary_acc y out_path para que se guarde el PNG final del heatmap
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
