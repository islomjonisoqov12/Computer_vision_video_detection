"""ByteTrack integration via Supervision, plus per-ID motion history."""
import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import supervision as sv


@dataclass
class Track:
    """Everything observed about one tracked object, in source-frame pixels."""

    track_id: int
    times: list[float] = field(default_factory=list)
    boxes: list[np.ndarray] = field(default_factory=list)  # xyxy
    class_votes: Counter = field(default_factory=Counter)

    def add(self, t: float, xyxy: np.ndarray, class_id: int) -> None:
        self.times.append(t)
        self.boxes.append(xyxy)
        self.class_votes[class_id] += 1

    @property
    def class_id(self) -> int:
        # YOLO flickers between e.g. car/truck; the majority label is stable.
        return self.class_votes.most_common(1)[0][0]

    @property
    def first_seen(self) -> float:
        return self.times[0]

    @property
    def last_seen(self) -> float:
        return self.times[-1]

    @property
    def points(self) -> np.ndarray:
        """Bottom-centre of each box (ground contact point), shape (N, 2)."""
        b = np.asarray(self.boxes)
        return np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)

    def velocity(self, window: float = 1.0, at: float | None = None) -> tuple[float, float]:
        """Mean (vx, vy) in px/s over the `window` seconds ending at `at` (default: last seen)."""
        at = self.last_seen if at is None else at
        times = np.asarray(self.times)
        idx = np.nonzero((times >= at - window) & (times <= at))[0]
        if len(idx) < 2:
            return 0.0, 0.0
        i, j = idx[0], idx[-1]
        dt = times[j] - times[i]
        if dt <= 0:
            return 0.0, 0.0
        pts = self.points
        return tuple((pts[j] - pts[i]) / dt)

    def speed(self, window: float = 1.0, at: float | None = None) -> float:
        """Pixels per second."""
        return math.hypot(*self.velocity(window, at))

    def heading(self, window: float = 1.0, at: float | None = None) -> float | None:
        """Direction of travel in image degrees (0 = right, 90 = down), None if not moving."""
        vx, vy = self.velocity(window, at)
        if vx == 0 and vy == 0:
            return None
        return math.degrees(math.atan2(vy, vx)) % 360


class Tracker:
    def __init__(self, fps: float, lost_seconds: float = 2.0, min_consecutive: int = 2):
        self.fps = fps
        self.lost_seconds = lost_seconds
        # Detections in (0.1, 0.25) confidence feed ByteTrack's second association pass;
        # new tracks start only from >= 0.25 and must be seen on `min_consecutive` updates.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            minimum_consecutive_frames=min_consecutive,
            frame_rate=fps,
        )
        self.tracks: dict[int, Track] = {}

    def update(self, detections: sv.Detections, t: float, stride: int = 1) -> sv.Detections:
        """Assign persistent tracker_id values and record history.

        `stride` is the number of source frames since the previous update; the
        lost-track buffer is kept in seconds even though the stride varies.
        """
        self.tracker.max_time_lost = max(1, round(self.lost_seconds * self.fps / stride))
        tracked = self.tracker.update_with_detections(detections)
        for xyxy, class_id, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
            tid = int(tid)
            if tid not in self.tracks:
                self.tracks[tid] = Track(tid)
            self.tracks[tid].add(t, xyxy.copy(), int(class_id))
        return tracked

    def active(self, t: float, max_age: float = 0.5) -> list[Track]:
        """Tracks seen within `max_age` seconds before t."""
        return [tr for tr in self.tracks.values() if t - tr.last_seen <= max_age]

    def reset(self) -> None:
        self.tracker.reset()
        self.tracks.clear()
