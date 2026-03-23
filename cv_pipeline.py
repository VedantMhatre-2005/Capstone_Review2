"""
cv_pipeline.py — Computer Vision Pipeline for Traffic Feature Extraction
=========================================================================
Capstone Review 2 | Yash — CV + Data Module

Reads two intersection video feeds (via RTSP from mediamtx, or directly
from file), runs YOLOv8n detection + ByteTrack tracking on CUDA, and
produces real node/edge feature arrays for Vedant's GNN pipeline.

Node Features (6D per intersection):
  phi_flow     — vehicles/hr (from YOLO count in rolling window)
  phi_signal   — signal phase time (0-120 s, synthetic cycle)
  phi_type     — intersection type ID (config)
  phi_x        — normalized X map coord (config)
  phi_y        — normalized Y map coord (config)
  phi_deg      — number of connecting roads (config)

Edge Features (5D per road segment):
  epsilon_cap   — road capacity (config)
  epsilon_speed — avg vehicle speed km/h (from centroid tracking)
  epsilon_lanes — number of lanes (config)
  epsilon_len   — road length in metres (config)
  epsilon_type  — road type ID (config)

Usage:
  # Option 1: RTSP mode (requires launch_streams.bat running)
  python cv_pipeline.py --mode rtsp

  # Option 2: Direct file mode (no mediamtx needed)
  python cv_pipeline.py --mode file

  # Run for N seconds then exit (useful for testing)
  python cv_pipeline.py --mode file --duration 30
"""

import argparse
import csv
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import Thread, Event

import cv2
import numpy as np
import torch
from typing import Optional

# ============================================================================
# 0. DEPENDENCY CHECK
# ============================================================================

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "ultralytics not installed. Run: pip install ultralytics"
    )

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# --- Graph topology config (2-node graph for demo) ---
NODE_CONFIG = {
    0: {                       # Node 0 → video1.mp4 / rtsp node1
        "phi_type": 0,         # Intersection type: 4-way
        "phi_x":    0.25,      # Normalised map X
        "phi_y":    0.60,      # Normalised map Y
        "phi_deg":  4,         # Roads connected
        "video_source": "videos/video1.mp4",
        "rtsp_url":     "rtsp://localhost:8554/node1",
        "label":        "Intersection-A",
    },
    1: {                       # Node 1 → video2.mp4 / rtsp node2
        "phi_type": 1,         # Intersection type: T-junction
        "phi_x":    0.70,
        "phi_y":    0.45,
        "phi_deg":  3,
        "video_source": "videos/video2.mp4",
        "rtsp_url":     "rtsp://localhost:8554/node2",
        "label":        "Intersection-B",
    },
}

EDGE_CONFIG = {
    (0, 1): {                  # Road from node 0 → node 1
        "epsilon_cap":   600,  # Max vehicles the road can hold
        "epsilon_lanes": 2,    # Number of lanes
        "epsilon_len":   250,  # Physical length in metres
        "epsilon_type":  0,    # Local road
    },
    (1, 0): {                  # Road from node 1 → node 0
        "epsilon_cap":   600,
        "epsilon_lanes": 2,
        "epsilon_len":   250,
        "epsilon_type":  0,
    },
}

# --- YOLO & tracking ---
YOLO_MODEL      = "yolov8n.pt"          # Fastest model, ~150 FPS on RTX 4060
VEHICLE_CLASSES = {2, 3, 5, 7}          # car, motorcycle, bus, truck (COCO)
CONF_THRESHOLD  = 0.40
IOU_THRESHOLD   = 0.45
TRACKER         = "bytetrack.yaml"      # Built into ultralytics

# --- Feature computation ---
FLOW_WINDOW_SEC    = 60     # Rolling window for vehicle count → veh/hr
SIGNAL_CYCLE_SEC   = 120    # Traffic light cycle duration (seconds)
PX_PER_METRE       = 8.0    # Calibration: ~8 pixels = 1 metre (adjust per cam)
FPS_DEFAULT        = 25.0   # Fallback FPS if stream metadata unavailable

# --- Output ---
OUTPUT_DIR = Path("outputs")
NODE_CSV   = OUTPUT_DIR / "node_features_live.csv"
EDGE_CSV   = OUTPUT_DIR / "edge_features_live.csv"
GNN_CSV    = OUTPUT_DIR / "training_dataset_live.csv"

# Write interval in seconds (how often to flush CSVs)
WRITE_INTERVAL_SEC = 5

NODE_CSV_HEADERS = [
    "timestamp", "node_id", "label",
    "phi_flow", "phi_signal", "phi_type", "phi_x", "phi_y", "phi_deg",
]
EDGE_CSV_HEADERS = [
    "timestamp", "src", "dst",
    "epsilon_cap", "epsilon_speed", "epsilon_lanes", "epsilon_len", "epsilon_type",
]
GNN_CSV_HEADERS = [
    "flow", "signal", "type", "x", "y", "degree", "congestion_label",
]

# ============================================================================
# 2. HELPER UTILITIES
# ============================================================================

def select_device() -> str:
    """Return 'cuda' if a CUDA-capable GPU is available, else 'cpu'."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[CV] ✓ Using device: cuda ({name})")
        return "cuda"
    print("[CV] ⚠ CUDA not available — falling back to CPU")
    return "cpu"


def ensure_model(model_name: str = YOLO_MODEL) -> YOLO:
    """Load (and auto-download if missing) a YOLOv8 model."""
    print(f"[CV] Loading model: {model_name}")
    model = YOLO(model_name)
    model.info(verbose=False)
    return model


def open_stream(source: str, use_tcp: bool = True) -> cv2.VideoCapture:
    """
    Open an OpenCV VideoCapture for an RTSP URL or file path.
    Uses TCP transport for RTSP to avoid packet loss on localhost.
    """
    if source.startswith("rtsp://") and use_tcp:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"[CV] ✗ Cannot open stream: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[CV] ✓ Stream opened: {source}  [{w}×{h} @ {fps:.1f} FPS]")
    return cap, fps


def centroid(box) -> tuple[float, float]:
    """Return (cx, cy) of an xyxy bounding box."""
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def px_speed_to_kmh(px_delta: float, fps: float) -> float:
    """
    Convert pixel displacement between frames to km/h.
      px_delta  — Euclidean centroid shift in pixels (single frame)
      fps       — frames per second of the stream
    """
    metres_per_frame = px_delta / PX_PER_METRE
    metres_per_sec   = metres_per_frame * fps
    return metres_per_sec * 3.6  # m/s → km/h


def flow_to_congestion(flow_veh_hr: float, capacity: float) -> float:
    """
    Simple congestion label: flow / capacity clamped to [0, 1].
    This is the synthetic target (y) fed to the GNN.
    """
    return min(flow_veh_hr / max(capacity, 1.0), 1.0)


# ============================================================================
# 3. PER-NODE PROCESSOR
# ============================================================================

class NodeProcessor:
    """
    Processes a single intersection's video stream.
    Runs YOLO + ByteTrack on every frame and maintains rolling feature state.
    """

    def __init__(self, node_id: int, source: str, model: YOLO, device: str, fps: float):
        self.node_id  = node_id
        self.cfg      = NODE_CONFIG[node_id]
        self.model    = model
        self.device   = device
        self.fps      = fps

        # Rolling window: timestamps of detected vehicle events (for flow)
        self._event_times = deque()  # type: deque

        # Speed tracking: {track_id: (cx, cy, timestamp)}
        self._prev_positions = {}  # type: dict
        self._speed_samples = deque(maxlen=100)  # type: deque

        # Signal phase state
        self._signal_start = time.time()

        # Latest computed features
        self.phi_flow    = 0.0
        self._phi_signal = 0.0  # backing field initialised before property is used
        self.epsilon_speed = 30.0   # km/h default

    # ------------------------------------------------------------------
    # Feature accessors
    # ------------------------------------------------------------------

    @property
    def phi_signal(self):
        return self._phi_signal

    @phi_signal.setter
    def phi_signal(self, v):
        self._phi_signal = float(v)

    def get_node_features(self) -> list:
        """Return 6D node feature vector [flow, signal, type, x, y, deg]."""
        elapsed = (time.time() - self._signal_start) % SIGNAL_CYCLE_SEC
        self._phi_signal = elapsed
        return [
            self.phi_flow,
            self._phi_signal,
            self.cfg["phi_type"],
            self.cfg["phi_x"],
            self.cfg["phi_y"],
            self.cfg["phi_deg"],
        ]

    def get_edge_speed(self) -> float:
        """Mean speed over the last 100 samples (km/h)."""
        if len(self._speed_samples) == 0:
            return 30.0
        return float(np.mean(self._speed_samples))

    # ------------------------------------------------------------------
    # Core: process one decoded frame
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray):
        """
        Run YOLOv8n + ByteTrack on a single frame.
        Updates phi_flow and epsilon_speed in-place.
        """
        now = time.time()

        # Run detection + tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=list(VEHICLE_CLASSES),
            tracker=TRACKER,
            device=self.device,
            verbose=False,
        )

        if results is None or len(results) == 0:
            return

        result = results[0]

        # Gather detections
        boxes  = result.boxes
        if boxes is None or len(boxes) == 0:
            self._update_flow(now, count=0)
            return

        track_ids = boxes.id  # May be None if no tracked objects
        xyxy_all  = boxes.xyxy.cpu().numpy()

        n_vehicles = len(xyxy_all)
        self._update_flow(now, count=n_vehicles)

        if track_ids is not None:
            ids = track_ids.cpu().numpy().astype(int)
            for i, tid in enumerate(ids):
                cx, cy = centroid(xyxy_all[i])
                if tid in self._prev_positions:
                    px, py, pt = self._prev_positions[tid]
                    dt = now - pt
                    if dt > 0:
                        dist_px = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                        speed   = px_speed_to_kmh(dist_px / (dt * self.fps), self.fps)
                        # Sanity filter: 0-150 km/h
                        if 0 < speed < 150:
                            self._speed_samples.append(speed)
                self._prev_positions[tid] = (cx, cy, now)

        # Prune stale track IDs (not seen in 5 seconds)
        if track_ids is not None:
            active = set(track_ids.cpu().numpy().astype(int))
        else:
            active = set()
        stale = [k for k in self._prev_positions if k not in active and
                 (now - self._prev_positions[k][2]) > 5.0]
        for k in stale:
            del self._prev_positions[k]

    # ------------------------------------------------------------------
    # Rolling flow computation
    # ------------------------------------------------------------------

    def _update_flow(self, now: float, count: int):
        """
        Add `count` vehicle events at timestamp `now` and recompute
        phi_flow as vehicles-per-hour using a FLOW_WINDOW_SEC window.
        """
        for _ in range(count):
            self._event_times.append(now)

        # Drop events outside the window
        cutoff = now - FLOW_WINDOW_SEC
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()

        # Scale count in window to veh/hr
        elapsed = min(now - (self._event_times[0] if self._event_times else now),
                      FLOW_WINDOW_SEC)
        if elapsed > 0:
            self.phi_flow = len(self._event_times) * (3600.0 / elapsed)
        else:
            self.phi_flow = 0.0


# ============================================================================
# 4. STREAM READER THREAD
# ============================================================================

class StreamThread(Thread):
    """
    Background thread: reads frames from a VideoCapture and calls
    processor.process_frame() on each frame.
    """

    def __init__(self, node_id, source, processor, stop_event):
        # type: (int, str, NodeProcessor, Event) -> None
        super().__init__(daemon=True, name=f"stream-node{node_id}")
        self.node_id   = node_id
        self.source    = source
        self.processor = processor
        self.stop_event = stop_event
        self._error = None  # type: Optional[str]

    def run(self):
        retry_delay = 3  # seconds before reconnecting on failure
        while not self.stop_event.is_set():
            try:
                cap, fps = open_stream(self.source)
                self.processor.fps = fps
                frame_count = 0

                while not self.stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        # File ended → loop back (fallback for direct file mode)
                        if not self.source.startswith("rtsp://"):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        # RTSP stream lost → reconnect
                        print(f"[Node{self.node_id}] Stream ended — reconnecting in {retry_delay}s")
                        break

                    # Process every frame (YOLO is fast enough on RTX 4060)
                    self.processor.process_frame(frame)
                    frame_count = frame_count + 1

                cap.release()

            except RuntimeError as e:
                self._error = str(e)
                print(f"[Node{self.node_id}] ✗ {e} — retrying in {retry_delay}s")

            if not self.stop_event.is_set():
                time.sleep(retry_delay)

        print(f"[Node{self.node_id}] Stream thread stopped.")


# ============================================================================
# 5. CSV WRITER
# ============================================================================

class FeatureWriter:
    """Manages periodic writes of node/edge feature CSVs."""

    def __init__(self):
        OUTPUT_DIR.mkdir(exist_ok=True)
        self._init_csvs()

    def _init_csvs(self):
        for path, headers in [
            (NODE_CSV, NODE_CSV_HEADERS),
            (EDGE_CSV, EDGE_CSV_HEADERS),
            (GNN_CSV, GNN_CSV_HEADERS),
        ]:
            # Always overwrite on fresh run
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        print(f"[CV] ✓ Output CSVs initialised in '{OUTPUT_DIR}/'")

    def write(self, processors: dict[int, NodeProcessor]):
        """Write one snapshot row per node and one per edge."""
        ts = datetime.now().isoformat(timespec="seconds")

        # ---- Node features ----
        node_rows = []
        for node_id, proc in processors.items():
            feats = proc.get_node_features()
            cfg   = NODE_CONFIG[node_id]
            row = [ts, node_id, cfg["label"]] + [f"{v:.4f}" for v in feats]
            node_rows.append(row)

        with open(NODE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(node_rows)

        # ---- Edge features ----
        edge_rows   = []
        for (src, dst), ecfg in EDGE_CONFIG.items():
            # Edge speed: mean of the two endpoint speeds
            speed = (processors[src].get_edge_speed() +
                     processors.get(dst, processors[src]).get_edge_speed()) / 2.0
            row = [
                ts, src, dst,
                ecfg["epsilon_cap"],
                f"{speed:.4f}",
                ecfg["epsilon_lanes"],
                ecfg["epsilon_len"],
                ecfg["epsilon_type"],
            ]
            edge_rows.append(row)

        with open(EDGE_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(edge_rows)

        # ---- GNN-compatible training dataset ----
        gnn_rows = []
        for node_id, proc in processors.items():
            feats = proc.get_node_features()
            cap   = EDGE_CONFIG.get((node_id, 1 - node_id), {}).get("epsilon_cap", 600)
            congestion = flow_to_congestion(feats[0], cap)
            gnn_row = [f"{v:.6f}" for v in feats] + [f"{congestion:.6f}"]
            gnn_rows.append(gnn_row)

        with open(GNN_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(gnn_rows)

        # Console summary
        for node_id, proc in processors.items():
            print(
                f"  [Node{node_id}] flow={proc.phi_flow:6.0f} veh/hr  "
                f"signal={proc._phi_signal:5.1f}s  "
                f"speed={proc.get_edge_speed():5.1f} km/h  "
                f"[{ts}]"
            )


# ============================================================================
# 6. MAIN PIPELINE
# ============================================================================

def build_sources(mode):
    # type: (str) -> dict
    """Return {node_id: source_string} based on selected mode."""
    sources = {}  # type: dict
    for node_id, cfg in NODE_CONFIG.items():
        if mode == "rtsp":
            sources[node_id] = str(cfg["rtsp_url"])
        else:
            sources[node_id] = str(Path(__file__).parent / cfg["video_source"])
    return sources


def run_pipeline(mode="file", duration=None):
    # type: (str, Optional[float]) -> None
    print("=" * 70)
    print("  CV Pipeline — Traffic Feature Extraction")
    print(f"  Mode : {mode.upper()}")
    print(f"  Model: {YOLO_MODEL}  |  Tracker: ByteTrack")
    print("=" * 70)

    device = select_device()
    model  = ensure_model(YOLO_MODEL)

    sources = build_sources(mode)

    if mode == "rtsp":
        print("\n[CV] Waiting 3 s for RTSP streams to stabilise...")
        time.sleep(3)

    # ---- Initialise processors ----
    processors = {}  # type: dict
    for node_id, src in sources.items():
        print(f"\n[CV] Initialising Node {node_id}: {src}")
        # Probe FPS from file/stream
        try:
            cap, fps = open_stream(str(src))
            cap.release()
        except RuntimeError:
            fps = FPS_DEFAULT
            print(f"[Node{node_id}] Could not probe stream, using {fps} FPS default")
        processors[node_id] = NodeProcessor(node_id, str(src), model, device, fps)

    # ---- Start stream threads ----
    stop_event = Event()
    threads = []  # type: list
    for node_id, src in sources.items():
        t = StreamThread(node_id, str(src), processors[node_id], stop_event)
        t.start()
        threads.append(t)

    writer = FeatureWriter()

    print(f"\n[CV] Pipeline running. Writing features every {WRITE_INTERVAL_SEC}s")
    print( "[CV] Press Ctrl+C to stop.\n")

    start_time = time.time()
    try:
        while True:
            time.sleep(WRITE_INTERVAL_SEC)
            writer.write(processors)

            if duration is not None and (time.time() - start_time) >= duration:
                print(f"\n[CV] Duration limit reached ({duration}s). Stopping.")
                break

    except KeyboardInterrupt:
        print("\n[CV] Interrupted by user.")

    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=5)

    print("\n" + "=" * 70)
    print("  CV Pipeline Complete")
    print(f"  Node features  → {NODE_CSV}")
    print(f"  Edge features  → {EDGE_CSV}")
    print(f"  GNN dataset    → {GNN_CSV}")
    print("=" * 70)


# ============================================================================
# 7. CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CV Pipeline — YOLOv8n RTSP Traffic Feature Extractor"
    )
    parser.add_argument(
        "--mode",
        choices=["rtsp", "file"],
        default="file",
        help=(
            "rtsp: read from mediamtx RTSP streams (launch_streams.bat must be running). "
            "file: read video files directly (no mediamtx needed). "
            "Default: file"
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stop after this many seconds (useful for testing). Default: run forever.",
    )
    args = parser.parse_args()

    run_pipeline(mode=args.mode, duration=args.duration)
