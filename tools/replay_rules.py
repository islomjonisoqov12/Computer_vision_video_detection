"""Dev tool: run the rule engine on cached tracks (see cache_tracks.py) in seconds.

    python tools/replay_rules.py samples/C3896.MP4 [--snapshots OUT_DIR]

--snapshots writes one annotated frame per event (mid-segment) for eyeballing.
"""
import argparse
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.rules import RuleEngine, Scene, match_scene  # noqa: E402
from src.tracker import Tracker  # noqa: E402
from tools.cache_tracks import cache_path  # noqa: E402


def load_engine(video: str) -> tuple[RuleEngine, dict, np.ndarray]:
    with open(cache_path(video), "rb") as f:
        data = pickle.load(f)
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    tracker = Tracker(fps=data["fps"])
    tracker.tracks = data["tracks"]
    rules = RuleEngine(tracker, fps=data["fps"])
    rules.frame_size = (frame.shape[1], frame.shape[0])
    rules.scene = match_scene(frame)
    rules.signal_samples = data.get("signal_samples", {})
    return rules, data, frame


def draw_scene(img: np.ndarray, scene: Scene | None) -> None:
    if not scene:
        return
    for _, poly, _ in scene.crosswalks:
        cv2.polylines(img, [poly.vertices.astype(int)], True, (255, 255, 0), 4)
    for sl in scene.stop_lines:
        cv2.line(img, tuple(sl.a.astype(int)), tuple(sl.b.astype(int)), (0, 0, 255), 6)
        mid = ((sl.a + sl.b) / 2).astype(int)
        cv2.arrowedLine(img, tuple(mid), tuple((mid + sl.normal * 120).astype(int)), (0, 0, 255), 5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--snapshots")
    ap.add_argument("--only", help="comma-separated labels to snapshot")
    args = ap.parse_args()

    rules, data, _ = load_engine(args.video)
    duration = data["n_frames"] / data["fps"]
    t0 = time.perf_counter()
    events = rules.finalize(duration)
    print(f"scene={rules.scene.name if rules.scene else None}  "
          f"red phases={ {k: [(round(s), round(e)) for s, e in v] for k, v in rules.red.items()} }")
    for line in rules.log:
        print("  log:", line)
    print(f"{len(events)} events in {time.perf_counter() - t0:.1f}s: {dict(Counter(e[2] for e in events))}")
    for label, hits in rules.hits.items():
        for s, e, note in sorted(hits):
            print(f"  {label:<18} {s:7.2f} - {e:7.2f}  {note}")

    if args.snapshots:
        out = Path(args.snapshots)
        out.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(args.video)
        names = data["names"]
        # one snapshot per output event, highlighting every track whose hit it contains
        shots = []
        for s, e, label in events:
            if args.only and label not in args.only.split(","):
                continue
            notes = [n for hs, he, n in rules.hits[label] if hs < e and he > s]
            shots.append((label, s, e, "; ".join(dict.fromkeys(notes))))
        for k, (label, s, e, note) in enumerate(shots):
            t = (s + e) / 2 if e - s < 6 else s + 2.0
            involved = {int(x) for x in re.findall(r"#(\d+)", note)}
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * data["fps"]))
            ok, img = cap.read()
            if not ok:
                continue
            draw_scene(img, rules.scene)
            for tr in rules.tracker.tracks.values():
                times = np.asarray(tr.times)
                i = np.searchsorted(times, t)
                if i >= len(times) or abs(times[i] - t) > 0.3:
                    continue
                x1, y1, x2, y2 = tr.boxes[i].astype(int)
                col, th = ((0, 0, 255), 8) if tr.track_id in involved else ((0, 255, 0), 3)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
                cv2.putText(img, f"#{tr.track_id} {names[tr.class_id]}", (x1, y1 - 8), 0, 1.2, col, 3)
                if tr.track_id in involved:  # 3 s trail up to now
                    pts = tr.points[(times >= t - 3) & (times <= t)].astype(int)
                    cv2.polylines(img, [pts], False, (0, 0, 255), 6)
            cv2.putText(img, f"{label} {s:.1f}-{e:.1f}s {note}", (40, 120), 0, 2.2, (0, 0, 255), 7)
            cv2.imwrite(str(out / f"{k:03d}_{label}_{s:.0f}.jpg"), cv2.resize(img, (1920, 1080)))
        print(f"snapshots -> {out}")


if __name__ == "__main__":
    main()
