"""Business logic translating tracks into the 14 event classes.

Everything that depends on the camera view lives in a scene file (scenes/*.json:
crosswalks, stop lines, signal heads), matched to the video by its first frame.
Everything else is learned from the video's own tracks in finalize(): which
image cells are road (vehicles drive there) and the normal flow direction per
cell. Rules that need a scene element that is missing simply emit nothing.

Speeds are in object heights per second ("sizes/s") so the same thresholds
work near and far from the camera.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from matplotlib.path import Path as Polygon

from src.detector import BICYCLE, BUS, MOTORCYCLE, PERSON, VEHICLES
from src.tracker import Track, Tracker

# Must match evaluate.OFFICIAL_CLASSES; events with any other label are dropped.
EVENT_CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

SCENES_DIR = Path(__file__).resolve().parent.parent / "scenes"

VELOCITY_WINDOW = 1.0   # s, centred window for per-sample velocity
STOP_SPEED = 0.15       # sizes/s: below this an object is stationary
MOVE_SPEED = 0.6        # sizes/s: above this an object is clearly moving
GRID = 32               # road / flow map resolution (cells per image width)
MIN_ROAD_TRACKS = 4     # distinct moving vehicles needed to call a cell road

STOPPED_MIN_SEC = 10.0
QUEUE_DISCHARGE_SEC = 20.0  # a stop ending this soon after red->green is a signal queue
AMBER_GRACE_SEC = 1.0       # crossing this soon after red onset is not a violation
JAYWALK_MIN_SEC = 1.0
CROSSWALK_ENTRY_GRACE_SEC = 3.0
WRONG_WAY_MIN_SEC = 1.5
CONGESTION_MIN_SEC = 30.0
CONGESTION_MIN_SLOW = 8


# ----------------------------------------------------------------------------
# scene
# ----------------------------------------------------------------------------
def _thumb(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


@dataclass
class StopLine:
    name: str
    a: np.ndarray        # endpoints, frame px
    b: np.ndarray
    normal: np.ndarray   # unit, points in the direction of legal travel across the line
    signal: str | None


class Scene:
    def __init__(self, cfg: dict, frame_size: tuple[int, int]):
        w, h = frame_size
        sx, sy = w / cfg["size"][0], h / cfg["size"][1]
        scale = lambda pts: np.asarray(pts, dtype=float) * [sx, sy]  # noqa: E731

        self.name = cfg["name"]
        self.signals = {
            name: [(scale(np.reshape(head["roi"], (2, 2))).astype(int).ravel(), head.get("green_means", "green"))
                   for head in sig["heads"]]
            for name, sig in cfg.get("signals", {}).items()
        }
        self.crosswalks = [
            (cw["name"], Polygon(scale(cw["polygon"])), cw.get("walk_when"))
            for cw in cfg.get("crosswalks", [])
        ]
        self.stop_lines = []
        for sl in cfg.get("stop_lines", []):
            a, b = scale(sl["line"])
            d = (b - a) / np.linalg.norm(b - a)
            normal = np.array([-d[1], d[0]])
            if normal @ np.asarray(sl["direction"], dtype=float) < 0:
                normal = -normal
            self.stop_lines.append(StopLine(sl["name"], a, b, normal, sl.get("signal")))
        self.solid_lines = [scale(pl) for pl in cfg.get("solid_lines", [])]
        self.forbidden_turns = [
            (ft["name"], Polygon(scale(ft["from"])), Polygon(scale(ft["to"])))
            for ft in cfg.get("forbidden_turns", [])
        ]
        self.no_stopping = [Polygon(scale(p)) for p in cfg.get("ignore_stopped", [])]

    def read_signal(self, frame: np.ndarray, name: str) -> str | None:
        """'red' / 'green' / None from the lit lamps of all heads of this signal.

        A head with green_means='red' is in opposite phase (e.g. a pedestrian
        head facing the camera across the controlled road).
        """
        red = green = 0
        for (x1, y1, x2, y2), green_means in self.signals[name]:
            hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
            hh, ss, vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            r = int((((hh <= 10) | (hh >= 165)) & (ss >= 80) & (vv >= 70)).sum())
            g = int(((hh >= 65) & (hh <= 95) & (ss >= 100) & (vv >= 100)).sum())
            if green_means == "red":
                r, g = g, r
            red, green = red + r, green + g
        if red >= 10 and red > 2 * green:
            return "red"
        if green >= 10 and green > 2 * red:
            return "green"
        return None


def match_scene(frame: np.ndarray, min_corr: float = 0.6) -> Scene | None:
    """Pick the scene whose reference thumbnail best matches this frame."""
    best, best_corr = None, min_corr
    thumb = _thumb(frame).astype(np.float32)
    for path in sorted(SCENES_DIR.glob("*.json")):
        cfg = json.loads(path.read_text())
        ref = cv2.imread(str(SCENES_DIR / cfg["reference"]), cv2.IMREAD_GRAYSCALE)
        if ref is None:
            continue
        corr = float(cv2.matchTemplate(thumb, ref.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
        if corr > best_corr:
            best, best_corr = cfg, corr
    return Scene(best, (frame.shape[1], frame.shape[0])) if best else None


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def runs(mask: np.ndarray, t: np.ndarray, max_gap: float = 0.5, min_len: float = 0.0) -> list[tuple[float, float]]:
    """Maximal [start, end] time intervals where mask is True, bridging gaps < max_gap s."""
    out: list[list[float]] = []
    for ti, m in zip(t, mask):
        if not m:
            continue
        if out and ti - out[-1][1] <= max_gap:
            out[-1][1] = ti
        else:
            out.append([ti, ti])
    return [(s, e) for s, e in out if e - s >= min_len]


def box_iou_pairs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise IoU of two (N, 4) xyxy arrays."""
    iw = np.clip(np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])  # noqa: E731
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def align(a: "Motion", b: "Motion", tol: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (i into a, j into b) of samples taken at (nearly) the same time."""
    if a.t[-1] < b.t[0] or b.t[-1] < a.t[0]:
        return np.empty(0, int), np.empty(0, int)
    j = np.clip(np.searchsorted(b.t, a.t), 1, len(b.t) - 1) if len(b.t) > 1 else np.zeros(len(a.t), int)
    if len(b.t) > 1:
        j = np.where(np.abs(b.t[j - 1] - a.t) < np.abs(b.t[j] - a.t), j - 1, j)
    ok = np.abs(b.t[j] - a.t) <= tol
    return np.nonzero(ok)[0], j[ok]


def merge_segments(segs: list[tuple], gap: float = 1.0) -> list[list[float]]:
    """Union of (start, end, *extra) segments; extras are dropped."""
    out: list[list[float]] = []
    for s, e, *_ in sorted(segs):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


class Motion:
    """Per-sample arrays for one track: time, box, foot point, size, velocity."""

    def __init__(self, track: Track):
        self.id = track.track_id
        self.cls = track.class_id
        self.t = np.asarray(track.times)
        self.box = np.asarray(track.boxes)
        self.foot = track.points
        self.size = np.maximum(self.box[:, 3] - self.box[:, 1], 1.0)
        n = len(self.t)
        lo = np.searchsorted(self.t, self.t - VELOCITY_WINDOW / 2)
        hi = np.clip(np.searchsorted(self.t, self.t + VELOCITY_WINDOW / 2, side="right") - 1, 0, n - 1)
        dt = self.t[hi] - self.t[lo]
        self.vel = np.zeros((n, 2))
        ok = dt > 0
        self.vel[ok] = (self.foot[hi[ok]] - self.foot[lo[ok]]) / dt[ok, None]
        self.nspeed = np.linalg.norm(self.vel, axis=1) / self.size

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLES

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    def at(self, t: float) -> int | None:
        """Index of the sample nearest to t, if within 0.5 s."""
        i = int(np.clip(np.searchsorted(self.t, t), 0, len(self.t) - 1))
        if i > 0 and abs(self.t[i - 1] - t) < abs(self.t[i] - t):
            i -= 1
        return i if abs(self.t[i] - t) <= 0.5 else None


# ----------------------------------------------------------------------------
# engine
# ----------------------------------------------------------------------------
class RuleEngine:
    def __init__(self, tracker: Tracker, fps: float):
        self.tracker = tracker  # tracker.tracks holds full per-ID history
        self.fps = fps
        self.scene: Scene | None = None
        self.frame_size: tuple[int, int] | None = None
        self.signal_samples: dict[str, list[tuple[float, str | None]]] = {}

    def update(self, t: float, tracks: sv.Detections, frame: np.ndarray) -> None:
        """Called after each processed frame with that frame's tracked detections."""
        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.scene = match_scene(frame)
            if self.scene:
                self.signal_samples = {name: [] for name in self.scene.signals}
        if self.scene:
            for name in self.scene.signals:
                self.signal_samples[name].append((t, self.scene.read_signal(frame, name)))

    # ---- signal timeline ---------------------------------------------------
    def _build_signals(self) -> None:
        """Smooth raw readings (1 s majority) into red intervals per signal."""
        self.red: dict[str, list[tuple[float, float]]] = {}
        for name, samples in self.signal_samples.items():
            if not samples:
                continue
            t = np.array([s[0] for s in samples])
            r = np.array([s[1] == "red" for s in samples], dtype=float)
            g = np.array([s[1] == "green" for s in samples], dtype=float)
            lo = np.searchsorted(t, t - 0.5)
            hi = np.searchsorted(t, t + 0.5, side="right")
            cr, cg = np.concatenate([[0], np.cumsum(r)]), np.concatenate([[0], np.cumsum(g)])
            red = (cr[hi] - cr[lo]) > (cg[hi] - cg[lo])
            self.red[name] = runs(red, t, max_gap=1.5, min_len=3.0)  # drop glare/occlusion blips

    def red_since(self, signal: str | None, t: float) -> float | None:
        """Seconds the signal has been red at time t, None if not red / unknown."""
        for s, e in self.red.get(signal, []):
            if s <= t <= e:
                return t - s
        return None

    def _red_left(self, signal: str | None, t: float) -> float:
        """Seconds until the current red phase ends (inf if not red)."""
        for s, e in self.red.get(signal, []):
            if s <= t <= e:
                return e - t
        return math.inf

    def _validate_stop_lines(self, vehicles: list[Motion]) -> None:
        """Unlink a stop line from its signal if traffic routinely crosses it on red:
        the signal head we read does not govern that approach."""
        for line in (self.scene.stop_lines if self.scene else []):
            if line.signal not in self.red:
                line.signal = None
                continue
            times = [tc for m in vehicles for tc, _ in self._crossings(m, line)]
            on_red = sum((self.red_since(line.signal, tc) or 0) >= AMBER_GRACE_SEC for tc in times)
            if len(times) >= 10 and on_red / len(times) > 0.2:
                self.log.append(f"stop line {line.name}: {on_red}/{len(times)} crossings on red -> signal ignored")
                line.signal = None

    def green_starts(self, signal: str | None) -> list[float]:
        return [e for _, e in self.red.get(signal, [])]

    # ---- learned maps --------------------------------------------------------
    def _cells(self, pts: np.ndarray) -> np.ndarray:
        w, h = self.frame_size
        cell = w / GRID
        cx = np.clip((pts[:, 0] / cell).astype(int), 0, GRID - 1)
        cy = np.clip((pts[:, 1] / cell).astype(int), 0, int(math.ceil(h / cell)) - 1)
        return cy * GRID + cx

    def _build_maps(self, motions: list[Motion]) -> None:
        w, h = self.frame_size
        n_cells = GRID * int(math.ceil(h / (w / GRID)))
        road_tracks = np.zeros(n_cells)
        flow = np.zeros((n_cells, 2))
        flow_n = np.zeros(n_cells)
        for m in motions:
            if not m.is_vehicle:
                continue
            moving = m.nspeed > MOVE_SPEED
            if not moving.any():
                continue
            cells = self._cells(m.foot[moving])
            unit = m.vel[moving] / np.linalg.norm(m.vel[moving], axis=1, keepdims=True)
            for c in np.unique(cells):  # one vote per track per cell
                road_tracks[c] += 1
                flow[c] += unit[cells == c].mean(axis=0)
                flow_n[c] += 1
        self.road = road_tracks >= MIN_ROAD_TRACKS
        # Road cells whose 8 neighbours are all road too: excludes kerbs, bus stops
        # and sidewalks that a coarse cell straddles.
        rows = len(self.road) // GRID
        grid = self.road.reshape(rows, GRID)
        padded = np.pad(grid, 1, constant_values=False)
        core = np.ones_like(grid)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                core &= padded[dy:dy + rows, dx:dx + GRID]
        self.road_core = core.ravel()
        mean = flow / np.maximum(flow_n, 1)[:, None]
        self.flow_consistency = np.linalg.norm(mean, axis=1)
        self.flow_dir = mean / np.maximum(self.flow_consistency, 1e-9)[:, None]
        self.flow_n = flow_n

    def on_road(self, pts: np.ndarray) -> np.ndarray:
        return self.road[self._cells(pts)]

    def in_crosswalk(self, pts: np.ndarray) -> np.ndarray:
        inside = np.zeros(len(pts), dtype=bool)
        for _, poly, _ in (self.scene.crosswalks if self.scene else []):
            inside |= poly.contains_points(pts, radius=self._pad())
        return inside

    def _pad(self) -> float:
        return self.frame_size[0] * 0.025  # people walk along the stripe edges; feet sit below them

    # ---- finalize --------------------------------------------------------------
    def finalize(self, duration: float) -> list:
        """Return [[start_sec, end_sec, label], ...] for the whole video.

        Same-class segments must not overlap and must satisfy 0 <= start < end <= duration.
        """
        if self.frame_size is None:
            return []
        motions = [Motion(tr) for tr in self.tracker.tracks.values() if len(tr.times) >= 3]
        self._build_signals()
        self._build_maps(motions)
        vehicles = [m for m in motions if m.is_vehicle]
        persons = [m for m in motions if m.is_person and not self._is_rider(m, motions)]
        self.log: list[str] = []
        self._validate_stop_lines(vehicles)

        # label -> [(start, end, note), ...]; notes name the tracks involved (kept for debugging)
        found: dict[str, list[tuple]] = {c: [] for c in EVENT_CLASSES}
        self.hits = found
        found["stopped_vehicle"] += self.stopped_vehicle(vehicles, motions)
        red_light, stop_line = self._line_events(vehicles)
        found["red_light"] += red_light
        found["stop_line"] += stop_line
        found["wrong_way"] += self.wrong_way(vehicles)
        found["illegal_u_turn"] += self.illegal_u_turn(vehicles)
        found["illegal_turn"] += self.illegal_turn(vehicles)
        found["solid_line_crossing"] += self.solid_line_crossing(vehicles)
        found["jaywalking"] += self.jaywalking(persons)
        found["failure_to_yield"] += self.failure_to_yield(vehicles, persons)
        acc, near = self.collisions(vehicles, persons)
        found["accident"] += acc
        found["near_miss"] += near
        found["congestion"] += self.congestion(vehicles, duration)
        # road_obstacle, fire_smoke: need classes the COCO detector does not have.

        events = []
        for label, segs in found.items():
            for s, e in merge_segments(segs):
                s, e = max(0.0, s), min(duration, e)
                if e > s:
                    events.append([round(s, 2), round(e, 2), label])
        return sorted(events)

    # ---- rules -------------------------------------------------------------------
    def _is_rider(self, p: Motion, motions: list[Motion]) -> bool:
        """A person whose box mostly sits on a bike/motorcycle track is a rider, not a pedestrian."""
        for m in motions:
            if m.cls not in (BICYCLE, MOTORCYCLE):
                continue
            i, j = align(p, m)
            if len(i) and (box_iou_pairs(p.box[i], m.box[j]) > 0.2).sum() >= 0.5 * len(p.t):
                return True
        return False

    def stopped_vehicle(self, vehicles: list[Motion], motions: list[Motion]) -> list:
        """Stationary >= 10 s on the road while traffic gets past it, and not waiting in a queue.

        A queue is recognised by the visible signal (stop mostly during red, released
        soon after green), by standing at the head of a stop line, or, for approaches
        whose signal we cannot see, by other vehicles standing next to it.
        """
        out = []
        for m in vehicles:
            for s, e in runs(m.nspeed < STOP_SPEED, m.t, max_gap=1.0, min_len=STOPPED_MIN_SEC):
                if self._explained_stop(m, s, e, vehicles):
                    continue
                if m.cls == BUS and e - s < 60:
                    continue  # dwell at a bus stop
                out.append((s, e, f"#{m.id} stationary {e - s:.0f}s"))
        return out

    def _explained_stop(self, m: Motion, s: float, e: float, vehicles: list[Motion]) -> bool:
        """True if a stop in [s, e] is ordinary: off-road, in a queue, or not blocking anyone."""
        idx = (m.t >= s) & (m.t <= e)
        pos = np.median(m.foot[idx], axis=0, keepdims=True)
        size = float(np.median(m.size[idx]))
        if not self.on_road(pos)[0]:
            return True  # parking bay, sidewalk
        if any(p.contains_points(pos)[0] for p in (self.scene.no_stopping if self.scene else [])):
            return True
        if self._signal_queue(s, e) or self._at_stop_line(pos[0], size):
            return True
        if self._standing_neighbours(m, s, e, pos[0], size, vehicles):
            return True
        return not self._traffic_passes(pos[0], size, s, e, m, vehicles)

    def _signal_queue(self, s: float, e: float) -> bool:
        """Mostly during red and released soon after it turned green."""
        for sig, reds in self.red.items():
            red_time = sum(max(0.0, min(e, re) - max(s, rs)) for rs, re in reds)
            released = any(0 <= e - g <= QUEUE_DISCHARGE_SEC for g in self.green_starts(sig))
            if released and red_time >= 0.5 * (e - s):
                return True
        return False

    def _at_stop_line(self, pos: np.ndarray, size: float) -> bool:
        """Within 4 sizes before any stop line: head of a signal queue."""
        for line in (self.scene.stop_lines if self.scene else []):
            seg = line.b - line.a
            side = (pos - line.a) @ line.normal / size
            along = (pos - line.a) @ seg / (seg @ seg)
            if -4.0 <= side <= 0.5 and -0.1 <= along <= 1.1:
                return True
        return False

    @staticmethod
    def _standing_neighbours(m: Motion, s: float, e: float, pos: np.ndarray, size: float,
                             vehicles: list[Motion]) -> bool:
        """Another vehicle stands within 3 sizes for at least half of [s, e]: a queue."""
        for o in vehicles:
            if o is m or o.t[-1] < s or o.t[0] > e:
                continue
            oi = np.nonzero((o.t >= s) & (o.t <= e))[0]
            if len(oi) < 3:
                continue
            ok = (np.linalg.norm(o.foot[oi] - pos, axis=1) < 3 * size) & (o.nspeed[oi] < 2 * STOP_SPEED)
            covered = ok.mean() * (o.t[oi[-1]] - o.t[oi[0]])
            if covered >= 0.5 * (e - s):
                return True
        return False

    @staticmethod
    def _traffic_passes(pos: np.ndarray, size: float, s: float, e: float, m: Motion,
                        vehicles: list[Motion]) -> bool:
        """Some other vehicle drives past within 4 sizes while m stands."""
        for o in vehicles:
            if o is m or o.t[-1] < s or o.t[0] > e:
                continue
            idx = (o.t >= s) & (o.t <= e) & (o.nspeed > MOVE_SPEED)
            if idx.any() and (np.linalg.norm(o.foot[idx] - pos, axis=1) < 4 * size).sum() >= 3:
                return True
        return False

    def _crossings(self, m: Motion, line: StopLine) -> list[tuple[float, int]]:
        """(time, index) where the foot point crosses the line in its legal direction."""
        rel = m.foot - line.a
        side = rel @ line.normal
        seg = line.b - line.a
        along = (rel @ seg) / (seg @ seg)
        out = []
        for i in range(1, len(m.t)):
            if side[i - 1] < 0 <= side[i] and -0.05 <= along[i] <= 1.05:
                f = -side[i - 1] / (side[i] - side[i - 1])
                out.append((m.t[i - 1] + f * (m.t[i] - m.t[i - 1]), i))
        return out

    def _line_events(self, vehicles: list[Motion]) -> tuple[list, list]:
        """Classify every signalled stop-line crossing near a red phase.

        red_light: crossed on red (after the amber grace) and drove on into the junction.
        stop_line: crossed shortly before or during red, then stopped past the line.
        Vehicles that crossed on green and got stuck beyond the line are neither.
        """
        red_light, stop_line = [], []
        for line in (self.scene.stop_lines if self.scene else []):
            if line.signal is None:
                continue
            seg = line.b - line.a
            for m in vehicles:
                side = (m.foot - line.a) @ line.normal / m.size
                along = (m.foot - line.a) @ seg / (seg @ seg)
                for tc, i in self._crossings(m, line):
                    red_for = self.red_since(line.signal, tc)
                    red_soon = self.red_since(line.signal, tc + 3.0) is not None
                    if red_for is None and not red_soon:
                        continue  # a green crossing
                    after = (m.t >= tc) & (m.t <= tc + 4.0)
                    went_on = (side[after] >= 1.5).any()
                    if went_on:
                        if (red_for is not None and red_for >= AMBER_GRACE_SEC
                                and self._red_left(line.signal, tc) >= AMBER_GRACE_SEC):
                            red_light.append((tc - 1.0, tc + 2.0, f"#{m.id} ran red on {line.name} ({red_for:.1f}s)"))
                        continue
                    red = np.array([self.red_since(line.signal, t) is not None for t in m.t])
                    waiting = ((m.t >= tc) & (side > 0) & (side < 2.0) & (along > -0.05) & (along < 1.05)
                               & (m.nspeed < STOP_SPEED) & red)
                    for s, e in runs(waiting, m.t, max_gap=1.0, min_len=2.0):
                        stop_line.append((s, e, f"#{m.id} stopped past {line.name}"))
                        break
        return red_light, stop_line

    def wrong_way(self, vehicles: list[Motion]) -> list:
        """Sustained travel (>= 1.5 s and >= 3 sizes) against the learned flow of a cell
        where traffic reliably goes one way."""
        out = []
        for m in vehicles:
            moving = m.nspeed > MOVE_SPEED
            if not moving.any() or self._teleports(m):
                continue
            cells = self._cells(m.foot)
            reliable = (self.flow_consistency[cells] > 0.8) & (self.flow_n[cells] >= 8)
            unit = m.vel / np.maximum(np.linalg.norm(m.vel, axis=1, keepdims=True), 1e-9)
            against = (unit * self.flow_dir[cells]).sum(axis=1) < -0.5
            for s, e in runs(moving & reliable & against, m.t, max_gap=0.7, min_len=WRONG_WAY_MIN_SEC):
                idx = np.nonzero((m.t >= s) & (m.t <= e))[0]
                travelled = np.linalg.norm(m.foot[idx[-1]] - m.foot[idx[0]]) / np.median(m.size[idx])
                if travelled >= 3.0:
                    out.append((s, e, f"#{m.id} {travelled:.0f} sizes against flow"))
        return out

    def illegal_u_turn(self, vehicles: list[Motion]) -> list:
        """The vehicle travels >= 3 sizes one way, then >= 3 sizes back the opposite way, on the road.

        Displacement (not per-sample heading) keeps box jitter on parked cars out;
        tracks that teleport are ID switches, not manoeuvres.
        """
        out = []
        for m in vehicles:
            if len(m.t) < 10 or self._teleports(m):
                continue
            moving = np.nonzero(m.nspeed > MOVE_SPEED)[0]
            if len(moving) < 6:
                continue
            u0 = m.vel[moving[:5]].mean(axis=0)
            if np.linalg.norm(u0) == 0:
                continue
            u0 /= np.linalg.norm(u0)
            size = float(np.median(m.size))
            proj = (m.foot - m.foot[moving[0]]) @ u0 / size
            k = int(np.argmax(proj))
            if proj[k] < 3 or proj[k] - proj[k:].min() < 3:
                continue
            back = k + int(np.argmax(proj[k:] <= proj[k] - 3))
            if not self.on_road(m.foot[k:k + 1])[0]:
                continue
            # manoeuvre = last point still heading forward .. first point well on the way back
            fwd = np.nonzero((m.vel[:k + 1] @ u0 > 0.5 * np.linalg.norm(m.vel[:k + 1], axis=1)) & (m.nspeed[:k + 1] > MOVE_SPEED))[0]
            start = m.t[fwd[-1]] if len(fwd) else m.t[k]
            if m.t[back] - start > 25.0:
                continue  # too slow for a manoeuvre: drifting box / ID mix-up
            out.append((start, m.t[back], f"#{m.id}"))
        return out

    @staticmethod
    def _teleports(m: Motion, max_speed: float = 8.0) -> bool:
        """Any step faster than max_speed sizes/s: the tracker swapped objects."""
        dt = np.maximum(np.diff(m.t), 1e-3)
        step = np.linalg.norm(np.diff(m.foot, axis=0), axis=1) / m.size[1:] / dt
        return bool((step > max_speed).any())

    def illegal_turn(self, vehicles: list[Motion]) -> list:
        out = []
        for _, src, dst in (self.scene.forbidden_turns if self.scene else []):
            for m in vehicles:
                a, b = src.contains_points(m.foot), dst.contains_points(m.foot)
                if a.any() and b.any() and np.argmax(a) < len(b) - 1 - np.argmax(b[::-1]):
                    out.append((m.t[np.argmax(a)], m.t[len(b) - 1 - np.argmax(b[::-1])], f"#{m.id}"))
        return out

    def solid_line_crossing(self, vehicles: list[Motion]) -> list:
        out = []
        for pl in (self.scene.solid_lines if self.scene else []):
            for a, b in zip(pl[:-1], pl[1:]):
                d = (b - a) / np.linalg.norm(b - a)
                normal = np.array([-d[1], d[0]])
                for line in (StopLine("solid", a, b, normal, None), StopLine("solid", a, b, -normal, None)):
                    for m in vehicles:
                        out += [(tc - 1.0, tc + 1.0, f"#{m.id}") for tc, _ in self._crossings(m, line)]
        return out

    def jaywalking(self, persons: list[Motion]) -> list:
        """On the carriageway outside a crosswalk, or entering a crosswalk against its signal."""
        out = []
        for p in persons:
            in_cw = self.in_crosswalk(p.foot)
            deep_road = self.road_core[self._cells(p.foot)]
            out += [(s, e, f"#{p.id} on road outside crosswalk")
                    for s, e in runs(deep_road & ~in_cw, p.t, max_gap=0.7, min_len=JAYWALK_MIN_SEC)]
            if not self.scene:
                continue
            for _, poly, walk_when in self.scene.crosswalks:
                if not walk_when:
                    continue
                inside = poly.contains_points(p.foot)
                for s, e in runs(inside & self.on_road(p.foot), p.t, max_gap=0.7, min_len=JAYWALK_MIN_SEC):
                    (sig, walk_state), = walk_when.items()
                    red_for = self.red_since(sig, s)
                    state = "red" if red_for is not None else ("green" if sig in self.red else None)
                    if state is None or state == walk_state:
                        continue
                    # entered while the walk phase was not showing (with grace after it ended)
                    since_change = red_for if red_for is not None else self._green_for(sig, s)
                    if since_change is not None and since_change > CROSSWALK_ENTRY_GRACE_SEC:
                        out.append((s, e, f"#{p.id} entered crosswalk against signal"))
        return out

    def _green_for(self, signal: str, t: float) -> float | None:
        ends = [g for g in self.green_starts(signal) if g <= t]
        return t - max(ends) if ends else None

    def failure_to_yield(self, vehicles: list[Motion], persons: list[Motion]) -> list:
        """Vehicle keeps driving through a crosswalk with a pedestrian on the carriageway
        part of it, directly in its path (<= 2 sizes ahead, within 1 size sideways)."""
        if not self.scene or not self.scene.crosswalks:
            return []
        out = []
        for m in vehicles:
            passing = self.in_crosswalk(m.foot) & (m.nspeed > MOVE_SPEED)
            if not passing.any():
                continue
            for p in persons:
                ii, jj = align(m, p)
                keep = passing[ii]
                ii, jj = ii[keep], jj[keep]
                if not len(ii):
                    continue
                ped_ok = self.in_crosswalk(p.foot[jj]) & self.on_road(p.foot[jj])
                heading = m.vel[ii] / np.linalg.norm(m.vel[ii], axis=1, keepdims=True)
                rel = (p.foot[jj] - m.foot[ii]) / m.size[ii, None]
                ahead = (rel * heading).sum(axis=1)
                lateral = np.abs(rel[:, 0] * heading[:, 1] - rel[:, 1] * heading[:, 0])
                hit = ped_ok & (ahead > 0) & (ahead < 2.0) & (lateral < 1.0)
                if hit.any():
                    t = m.t[ii[np.argmax(hit)]]
                    out.append((t - 1.0, t + 1.0, f"#{m.id} vs ped #{p.id}"))
        return out

    def collisions(self, vehicles: list[Motion], persons: list[Motion]) -> tuple[list, list]:
        """Two road users converge fast and one brakes hard at close range.

        accident:  they end up in contact and both stay stopped >= 10 s in a way
                   no queue explains (the scene is blocked).
        near_miss: same fast convergence and hard braking, no contact, then they move on.

        Queue arrivals are excluded because the follower's closing speed has already
        bled off before the boxes meet, and the resulting stop is queue-explained.
        """
        accidents, near = [], []
        # Far-away boxes are too small for their size/speed to be trusted, and a track
        # that teleports has swapped objects (typically behind a bus).
        min_size = 0.04 * self.frame_size[1]
        actors = [x for x in vehicles + persons if np.median(x.size) >= min_size and not self._teleports(x)]
        n_veh = sum(x.is_vehicle for x in actors)
        for ai, a in enumerate(actors[:n_veh]):
            for b in actors[ai + 1:]:
                ii, jj = align(a, b)
                if len(ii) < 3:
                    continue
                size = np.maximum(a.size[ii], b.size[jj])
                d = b.foot[jj] - a.foot[ii]
                dist = np.linalg.norm(d, axis=1)
                gap = dist / size
                if gap.min() > 1.0:
                    continue
                dn = d / np.maximum(dist, 1e-9)[:, None]
                closing = ((a.vel[ii] - b.vel[jj]) * dn).sum(axis=1) / size
                iou = box_iou_pairs(a.box[ii], b.box[jj])
                for k in np.nonzero(gap <= 1.0)[0]:
                    t = a.t[ii[k]]
                    win = (a.t[ii] >= t - 1.0) & (a.t[ii] <= t)
                    if closing[win].max() < 1.5:
                        continue  # not converging fast
                    brake = max(self._speed_at(x, t - 1.0) - self._speed_at(x, t + 0.5) for x in (a, b))
                    if brake < 1.5:
                        continue
                    if not (self._steady_box(a, t) and self._steady_box(b, t)):
                        continue  # occlusion is resizing a box: its "motion" is not real
                    touching = iou[k] >= 0.2 and gap[k] <= 0.5
                    if touching and self._blocked_after(a, b, t, vehicles):
                        accidents.append((t - 1.0, t + 5.0, f"#{a.id} + #{b.id}"))
                    elif not touching and max(self._speed_at(x, t + 4.0) for x in (a, b)) > MOVE_SPEED:
                        near.append((t - 1.0, t + 1.0, f"#{a.id} + #{b.id}"))
                    else:
                        continue
                    break  # one event per pair
        return accidents, near

    @staticmethod
    def _steady_box(m: Motion, t: float, window: float = 1.5, max_ratio: float = 1.25) -> bool:
        idx = (m.t >= t - window) & (m.t <= t + window)
        if idx.sum() < 3:
            return False
        w = m.box[idx, 2] - m.box[idx, 0]
        return m.size[idx].max() / m.size[idx].min() <= max_ratio and w.max() / max(w.min(), 1) <= max_ratio

    def _blocked_after(self, a: Motion, b: Motion, t: float, vehicles: list[Motion]) -> bool:
        for x in (a, b):
            if not x.is_vehicle:
                continue
            stops = [(s, e) for s, e in runs(x.nspeed < STOP_SPEED, x.t, max_gap=1.0, min_len=10.0) if s <= t + 2.0 <= e]
            if not stops or self._explained_stop(x, *stops[0], vehicles):
                return False
        return True

    @staticmethod
    def _speed_at(m: Motion, t: float) -> float:
        i = m.at(t)
        return float(m.nspeed[i]) if i is not None else 0.0

    def congestion(self, vehicles: list[Motion], duration: float) -> list:
        """Many slow vehicles on the road for a sustained period, outside red phases."""
        ts = np.arange(0.0, duration, 1.0)
        slow_frac, slow_n = np.zeros(len(ts)), np.zeros(len(ts))
        for k, t in enumerate(ts):
            n = slow = 0
            for m in vehicles:
                i = m.at(t)
                if i is None or not self.on_road(m.foot[i:i + 1])[0]:
                    continue
                n += 1
                slow += m.nspeed[i] < MOVE_SPEED / 2
            slow_n[k], slow_frac[k] = slow, slow / n if n else 0.0
        red = np.array([any(self.red_since(s, t) is not None for s in self.red) for t in ts])
        mask = (slow_n >= CONGESTION_MIN_SLOW) & (slow_frac >= 0.7) & ~red
        return [(s, e, "slow traffic") for s, e in runs(mask, ts, max_gap=3.0, min_len=CONGESTION_MIN_SEC)]
