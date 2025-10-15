# YOLO Real-time Modular App

Estructura:
```
yolo_realtime_app/
  main.py
  src/
    detector.py
    overlay.py
    utils.py
    video_io.py
  requirements.txt
```

## Uso

Instala dependencias (idealmente en un venv):
```
pip install -r requirements.txt
```

Ejecuta (cámara por defecto 0):
```
python main.py --model "C:/ruta/a/yolov12l-face.pt" --source 0 --imgsz 448 --conf 0.10 --iou 0.30 --agnostic-nms --save --out salida_detectada.mp4 --classes head
```

Para un archivo de video como fuente:
```
python main.py --model "C:/ruta/a/yolov12l-face.pt" --source "C:/ruta/video.mp4" --imgsz 448 --save
```

Teclas:
- **q**: salir
```

