"""Main entry point required by the organizers (loaded by run_submission.py).

Part A: detect_events(video_path) -> [[start_sec, end_sec, label], ...]
Part B: RiskEstimator.reset(meta) / .step(frame, t) -> accident risk in [0, 1]
"""
import time
from pathlib import Path

import cv2

from src.detector import Detector
from src.rules import EVENT_CLASSES, RuleEngine
from src.tracker import Tracker

CLASSES = list(EVENT_CLASSES)

# Harness budget is 3.0 x video duration for Part A + Part B together, and Part B
# decodes every frame again. Part A aims to finish within PART_A_TIME_FACTOR x duration
# and adapts the detection stride to the hardware it runs on.
PART_A_TIME_FACTOR = 1.0  # 1.2 left only ~60 s of slack on a CPU-only laptop
MIN_STRIDE, MAX_STRIDE = 2, 10

_detector = None


def get_detector() -> Detector:
    # Shared across videos so the model loads once per run.
    global _detector
    if _detector is None:
        _detector = Detector()
    return _detector


def detect_events(video_path: str) -> list:
    t0 = time.perf_counter()
    detector = get_detector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    time_target = PART_A_TIME_FACTOR * n_frames / fps

    stride = MIN_STRIDE
    tracker = Tracker(fps=fps)
    rules = RuleEngine(tracker, fps=fps)

    frame_idx, next_detect, last_detect = 0, 0, 0
    while True:
        if frame_idx < next_detect:
            # grab() skips the BGR conversion: ~2.5x cheaper than read() on 4K footage
            if not cap.grab():
                break
            frame_idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps
        tracks = tracker.update(detector.detect(frame), t, stride=max(1, frame_idx - last_detect))
        rules.update(t, tracks, frame)
        last_detect = frame_idx

        # Behind schedule -> skip more frames; comfortably ahead -> skip fewer.
        progress = (frame_idx + 1) / n_frames
        elapsed = time.perf_counter() - t0
        if elapsed > time_target * progress and stride < MAX_STRIDE:
            stride += 1
        elif elapsed < 0.8 * time_target * progress and stride > MIN_STRIDE:
            stride -= 1
        next_detect = frame_idx + stride
        frame_idx += 1
    cap.release()
    return rules.finalize(duration=frame_idx / fps)


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        self.meta = meta

    def step(self, frame, t: float) -> float:
        # TODO: accident anticipation score.
        return 0.0


if __name__ == "__main__":
    import sys
    print(Path(sys.argv[1]).name, detect_events(sys.argv[1]))
