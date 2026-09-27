"""YOLO model loading and per-frame inference."""
from pathlib import Path

import numpy as np
import supervision as sv
import torch
from ultralytics import YOLO

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"

# COCO class ids kept for traffic analysis.
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
VEHICLES = [CAR, MOTORCYCLE, BUS, TRUCK]
TRACKED_CLASSES = [PERSON, BICYCLE] + VEHICLES


class Detector:
    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.1,  # ByteTrack uses 0.1-0.25 detections to keep existing tracks alive
        imgsz: int = 960,  # pedestrians in 4K footage are too small at 640
        classes: list[int] | None = TRACKED_CLASSES,
        device: str | None = None,
    ):
        path = Path(weights)
        if not path.is_absolute():
            path = WEIGHTS_DIR / path
        # Ultralytics downloads official weights by name if the file is missing.
        self.model = YOLO(str(path) if path.exists() else path.name)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        # FP16 on GPU; `quantize` replaced the deprecated `half` flag in ultralytics 8.4.
        self.precision = {"quantize": 16} if self.device.startswith("cuda") else {}
        self.conf = conf
        self.imgsz = imgsz
        self.classes = classes

    @property
    def class_names(self) -> dict:
        return self.model.names

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Return boxes (xyxy, source-frame pixels), confidences and COCO class ids."""
        result = self.model(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            classes=self.classes,
            device=self.device,
            verbose=False,
            **self.precision,
        )[0]
        return sv.Detections.from_ultralytics(result)
