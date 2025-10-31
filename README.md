# YOLO Real-time Modular App
```markdown
# CROWD_DETECTION — YOLO Real-time Modular App

This repository contains a small modular application for running YOLO-based
object detection in real-time (camera or video file), producing:

- A detections video with bounding boxes and overlays.
- An optional heatmap video and heatmap PNG summarizing the accumulated detections.
- An optional CSV file with per-frame metrics.

Project structure (top-level):

```
CROWD_DETECTION/
  main.py                # CLI runner for local/desktop usage
  gradio_ui.py           # Lightweight Gradio front-end for uploads and camera
  src/                   # Core modules (detector, IO, utils, overlays, etc.)
    detector.py
    overlay.py
    video_io.py
    utils/               # utilities and pipeline helpers
  requirements.txt
  README.md
```

Getting started
-----------------
Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the CLI (default camera 0):

```powershell
python main.py --model "C:/path/to/best.pt" --source 0 --imgsz 448 --conf 0.10 --iou 0.30 --agnostic-nms --save --out output/detections.mp4 --classes head
```

Or use a local video file as input:

```powershell
python main.py --model "C:/path/to/best.pt" --source "C:/path/to/video.mp4" --imgsz 448 --save
```

Keyboard shortcuts (when the OpenCV window is shown):
- Esc or 'q': exit the application

Notes about the Gradio UI
-------------------------
Run the web UI with:

```powershell
python gradio_ui.py
```

The Gradio interface provides two tabs: Upload (send a video file) and Camera
(record a short clip). Outputs (detections video, heatmap video, heatmap PNG and
CSV) are available for preview (when supported by the browser) and download.

YOLO fine-tuning command
-------------------------
The models used in this project were fine-tuned using the Ultralytics YOLO
training command. An example command to fine-tune a detection model looks like:

```bash
# Example (adjust dataset, model and hyperparameters to your setup):
yolo task=detect mode=train model=yolov8n.pt data=./data/dataset.yaml epochs=50 imgsz=640 batch=16 lr=0.01
```

Replace `yolov8n.pt`, `./data/dataset.yaml`, and hyperparameters with the values
you used when fine-tuning. If you used a different YOLO version (Ultralytics
v4/v5 or custom training), adapt the command accordingly.

```

