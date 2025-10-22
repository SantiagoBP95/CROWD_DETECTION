import argparse
import os
import cv2
import numpy as np
import torch

from src.detector import Detector, DetectorConfig
from src.video_io import VideoReader, VideoWriter
from src.overlay import Overlay
from src.utils import FPSSmoother
from src.metrics import RunningCount  # asegúrate de tener este archivo
# heatmap final se genera desde summary_acc; no se necesita HeatmapAccumulator en main

def parse_args():
    p = argparse.ArgumentParser(description="Detección en tiempo real con YOLO (modular).")
    p.add_argument("--model", type=str, default="models/best.pt", help="Ruta al .pt (modelo YOLO).")
    p.add_argument("--source", type=str, default="data/input/input2.mp4", help="Fuente de video: índice (0,1,...) o ruta a archivo.")
    p.add_argument("--imgsz", type=int, default=896, help="Tamaño de entrada del modelo.")
    p.add_argument("--conf", type=float, default=0.10, help="Umbral de confianza.")
    p.add_argument("--iou", type=float, default=0.30, help="IoU para NMS.")
    p.add_argument("--device", type=str, default=None, help="Dispositivo: 'cpu', '0', 'cuda:0'...")
    p.add_argument("--save", action="store_true", default=True, help="Guardar salida a video.")
    p.add_argument("--out", type=str, default="output/salida_detectada.mp4", help="Ruta de salida de video.")
    p.add_argument("--classes", type=str, default="head", help="Lista separada por comas de clases permitidas (por nombre).")
    p.add_argument("--max-det", type=int, default=3000, help="Máximo de detecciones por imagen.")
    p.add_argument("--agnostic-nms", action="store_true", help="Usar NMS agnóstico (recomendado con una sola clase).")
    p.add_argument("--heatmap", action="store_true", help="Activar mapa de calor dinámico superpuesto.")
    p.add_argument("--heatmap-alpha", type=float, default=0.6, help="Opacidad del heatmap al superponer (0..1).")
    p.add_argument("--heatmap-decay", type=float, default=0.9, help="Decaimiento del acumulador por frame (0..1).")
    p.add_argument("--heatmap-kernel", type=int, default=31, help="Kernel para suavizado del heatmap (impar).")
    p.add_argument("--heatmap-video-kernel", type=int, default=None, help="Kernel para suavizado del heatmap cuando se genera el video por frame (impar). Si no se especifica, usa --heatmap-kernel.")
    return p.parse_args()

def parse_source(src_str: str):
    # Si es un número (sin comillas en CLI), úsalo como índice de cámara.
    try:
        return int(src_str)
    except ValueError:
        return src_str

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
    class_whitelist = [s.strip() for s in args.classes.split(",")] if args.classes else None

    # Resolver rutas relativas respecto al directorio del script
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Modelo: si se pasó una ruta relativa, interpretarla relativa al paquete
    model_path = args.model
    if not os.path.isabs(model_path):
        model_path = os.path.normpath(os.path.join(base_dir, model_path))
    args.model = model_path

    # Fuente de video: si es ruta relativa, interpretarla relativa al paquete
    if isinstance(args.source, str) and not args.source.isdigit():
        source_path = args.source
        if not os.path.isabs(source_path):
            source_path = os.path.normpath(os.path.join(base_dir, source_path))
        args.source = source_path
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

    # IO de video
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
    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.normpath(os.path.join(base_dir, out_path))
    args.out = out_path
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    writer = VideoWriter(out_path if args.save else None, (w, h), fps)
    # acumulador global para summary heatmap (para guardar al final)
        # acumulador global para summary heatmap (para generar video de heatmap)
    summary_acc = np.zeros((h, w), dtype=np.float32)
    # acumulador summary_acc recoge todos los centros; al final se guardará el heatmap
    heatmap_video_writer = None
    # Si estamos guardando el video de detección, también creamos siempre
    # el writer para el video del heatmap (no dependemos de --heatmap).
    if args.save:
        # crear writer para video del heatmap (misma carpeta que --out)
        heatmap_dir = out_dir if out_dir else base_dir
        heatmap_video_path = os.path.join(heatmap_dir, f"heatmap_{os.path.splitext(os.path.basename(out_path))[0]}.mp4")
        heatmap_video_writer = VideoWriter(heatmap_video_path, (w, h), fps)
        # contador de frames escritos y flag de advertencia
        heatmap_frames_written = 0
        heatmap_warned = False
        # comprobar que el writer interno se abrió correctamente
        try:
            ok = False
            if hasattr(heatmap_video_writer, 'writer') and heatmap_video_writer.writer is not None:
                # cv2.VideoWriter tiene isOpened()
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

    # Preferir el mapeo preprocesado en el detector si existe (consistente: dict de id->name)
    names = getattr(detector, 'names', detector.model.names)
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

            # Inferencia en cada frame (comportamiento original sin tracker)
            dets = detector.infer(frame)
            current_count = len(dets)
            people_counter.update(current_count)

            # Dibujo de cajas
            centers = []
            for (x1, y1, x2, y2), conf, cid in dets:
                name = names[cid] if cid in names else f"id{cid}"
                label = f"{name} {conf:.2f}"
                overlay.draw_box(frame, (x1, y1, x2, y2), label)
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                centers.append((cx, cy))

            # actualizar acumulador summary (sin decaimiento) -> sumar 1 en el pixel
            # aplicar decaimiento exponencial antes de sumar los centros del frame
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
                    summary_acc[iy, ix] += 1.0

            # Si se pidió, generar frame de heatmap a partir del acumulador y escribirlo
            if heatmap_video_writer is not None:
                # elegir kernel para video (si se pasó especifico usarlo)
                k_vid = args.heatmap_video_kernel if args.heatmap_video_kernel is not None else args.heatmap_kernel
                if k_vid is None:
                    k_vid = 1
                k = int(k_vid)
                if k < 1:
                    k = 1
                if k % 2 == 0:
                    k += 1
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
                # asegurar tamaño
                if heat_col.shape[:2] != (h, w):
                    heat_col = cv2.resize(heat_col, (w, h))
                # comprobar que el writer interno está abierto antes de escribir
                try:
                    opened = False
                    if hasattr(heatmap_video_writer, 'writer') and heatmap_video_writer.writer is not None:
                        try:
                            opened = bool(heatmap_video_writer.writer.isOpened())
                        except Exception:
                            opened = True
                    if opened:
                        heatmap_video_writer.write(heat_col)
                        heatmap_frames_written += 1
                    else:
                        if not heatmap_warned:
                            print(f"[WARN] heatmap_video_writer no está abierto; no se escribirán frames de heatmap.")
                            heatmap_warned = True
                except Exception:
                    if not heatmap_warned:
                        print(f"[WARN] error al intentar escribir frame del heatmap")
                        heatmap_warned = True

            # HUD: conteos y FPS
            overlay.draw_counts(frame, people_counter.current, people_counter.average, people_counter.max_count)
            overlay.draw_fps(frame, fps_sm.tick())

            # Mostrar y guardar el frame de detección normalmente
            cv2.imshow("Detección en tiempo real (YOLO)", frame)
            if writer is not None and writer.writer is not None:
                writer.write(frame)
            frame_idx += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # asegurar liberación de recursos y guardar resumen aunque el bucle se interrumpa
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
            # reportar número de frames escritos
            try:
                print(f"[INFO] Frames de heatmap escritos: {heatmap_frames_written}", flush=True)
                if os.path.exists(heatmap_video_path):
                    size = os.path.getsize(heatmap_video_path)
                    print(f"[INFO] Tamaño del archivo: {size} bytes -> {heatmap_video_path}", flush=True)
            except Exception:
                pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
