import argparse
import os
import cv2
import torch

from src.detector import Detector, DetectorConfig
from src.video_io import VideoReader, VideoWriter
from src.overlay import Overlay
from src.utils import FPSSmoother
from src.metrics import RunningCount  # asegúrate de tener este archivo

def parse_args():
    p = argparse.ArgumentParser(description="Detección en tiempo real con YOLO (modular).")
    p.add_argument("--model", type=str, default="models/best.pt", help="Ruta al .pt (modelo YOLO).")
    p.add_argument("--source", type=str, default="data/input/input2.mp4", help="Fuente de video: índice (0,1,...) o ruta a archivo.")
    p.add_argument("--imgsz", type=int, default=896, help="Tamaño de entrada del modelo.")
    p.add_argument("--conf", type=float, default=0.10, help="Umbral de confianza.")
    p.add_argument("--iou", type=float, default=0.30, help="IoU para NMS.")
    p.add_argument("--device", type=str, default=None, help="Dispositivo: 'cpu', '0', 'cuda:0'...")
    p.add_argument("--save", action="store_true", help="Guardar salida a video.")
    p.add_argument("--out", type=str, default="salida_detectada.mp4", help="Ruta de salida de video.")
    p.add_argument("--classes", type=str, default="head", help="Lista separada por comas de clases permitidas (por nombre).")
    p.add_argument("--max-det", type=int, default=3000, help="Máximo de detecciones por imagen.")
    p.add_argument("--agnostic-nms", action="store_true", help="Usar NMS agnóstico (recomendado con una sola clase).")
    return p.parse_args()

def parse_source(src_str: str):
    # Si es un número (sin comillas en CLI), úsalo como índice de cámara.
    try:
        return int(src_str)
    except ValueError:
        return src_str

def main():
    args = parse_args()

    # Dispositivo por defecto
    device = args.device if args.device is not None else (0 if torch.cuda.is_available() else "cpu")

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
    reader = VideoReader(source)
    w, h = reader.get_size()
    fps = reader.get_fps()
    writer = VideoWriter(args.out if args.save else None, (w, h), fps)

    names = detector.model.names
    print("Presiona 'q' para salir.")
    while True:
        frame = reader.read()
        if frame is None:
            break

        # Inferencia y conteo
        dets = detector.infer(frame)
        current_count = len(dets)
        people_counter.update(current_count)

        # Dibujo de cajas
        for (x1, y1, x2, y2), conf, cid in dets:
            name = names[cid] if cid in names else f"id{cid}"
            label = f"{name} {conf:.2f}"
            overlay.draw_box(frame, (x1, y1, x2, y2), label)

        # HUD: conteos y FPS
        overlay.draw_counts(frame, people_counter.current, people_counter.average, people_counter.max_count)
        overlay.draw_fps(frame, fps_sm.tick())

        # Mostrar / Guardar
        cv2.imshow("Detección en tiempo real (YOLO)", frame)
        writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    reader.release()
    writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
