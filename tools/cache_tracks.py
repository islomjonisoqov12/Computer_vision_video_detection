"""Dev tool: run detection + tracking (and signal reading) once and pickle the
result, so rules can be iterated on without re-running YOLO.

    python tools/cache_tracks.py samples/C3896.MP4 [--stride 3]
    python tools/cache_tracks.py samples/C3896.MP4 --signals-only   # add signals to an existing cache
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import supervision as sv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.detector import Detector  # noqa: E402
from src.rules import RuleEngine  # noqa: E402
from src.tracker import Tracker  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def cache_path(video: str) -> Path:
    return CACHE_DIR / (Path(video).name + ".tracks.pkl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--signals-only", action="store_true")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = None if args.signals_only else Detector()
    tracker = Tracker(fps=fps)
    rules = RuleEngine(tracker, fps=fps)
    t0, idx = time.perf_counter(), 0
    while True:
        if idx % args.stride:
            if not cap.grab():
                break
        else:
            ok, frame = cap.read()
            if not ok:
                break
            t = idx / fps
            tracks = tracker.update(detector.detect(frame), t, stride=args.stride) if detector else sv.Detections.empty()
            rules.update(t, tracks, frame)
        idx += 1
        if idx % 900 == 0:
            print(f"{idx}/{n_frames} frames, {time.perf_counter() - t0:.0f}s", flush=True)
    cap.release()

    out = cache_path(args.video)
    if args.signals_only:
        with open(out, "rb") as f:
            data = pickle.load(f)
    else:
        data = {"fps": fps, "n_frames": idx, "names": detector.class_names,
                "stride": args.stride, "tracks": tracker.tracks}
    data["signal_samples"] = rules.signal_samples
    data["scene"] = rules.scene.name if rules.scene else None
    CACHE_DIR.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(data, f)
    print(f"{len(data['tracks'])} tracks, scene={data['scene']} -> {out}")


if __name__ == "__main__":
    main()
