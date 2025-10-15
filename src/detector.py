from typing import Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO

class DetectorConfig:
    def __init__(
        self,
        model_path: str,
        img_size: int = 448,
        conf_thres: float = 0.10,
        iou_thres: float = 0.30,
        device: Union[int, str] = 0,
        class_whitelist: Optional[Iterable[str]] = None,
        max_det: int = 3000,
        agnostic_nms: bool = True
    ) -> None:
        self.model_path = model_path
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.class_whitelist = set([c.lower() for c in class_whitelist]) if class_whitelist else None
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms

class Detector:
    def __init__(self, cfg: DetectorConfig):
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        self.model.to(cfg.device)
        # Build allowed class id set
        names = self.model.names if hasattr(self.model, "names") else {}
        if isinstance(names, dict):
            self.names = names
        else:
            # Fallback: list to dict
            self.names = {i: n for i, n in enumerate(names)}
        self.allowed_ids: Optional[Set[int]] = None
        if self.cfg.class_whitelist:
            allowed = set()
            for cls_id, cls_name in self.names.items():
                if str(cls_name).lower() in self.cfg.class_whitelist:
                    allowed.add(cls_id)
            self.allowed_ids = allowed if allowed else set(self.names.keys())

    @torch.inference_mode()
    def infer(self, frame) -> List[Tuple[Tuple[int, int, int, int], float, int]]:
        """Returns list of (box_xyxy, conf, cls_id)."""
        res = self.model.predict(
            frame,
            imgsz=self.cfg.img_size,
            conf=self.cfg.conf_thres,
            iou=self.cfg.iou_thres,
            device=self.cfg.device,
            max_det=self.cfg.max_det,
            agnostic_nms=self.cfg.agnostic_nms,
            verbose=False
        )[0]

        outputs: List[Tuple[Tuple[int, int, int, int], float, int]] = []
        if res.boxes is not None:
            boxes = res.boxes.xyxy.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss  = res.boxes.cls.detach().cpu().numpy().astype(int)
            for (x1, y1, x2, y2), conf, cid in zip(boxes, confs, clss):
                if self.allowed_ids is not None and cid not in self.allowed_ids:
                    continue
                outputs.append(((int(x1), int(y1), int(x2), int(y2)), float(conf), int(cid)))
        return outputs
