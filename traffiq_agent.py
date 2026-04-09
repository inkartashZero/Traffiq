# ============================================================
# TRAFFIQ — Autonomous Driving Agent v2
# IET On-Campus JU × JUMPER
# ============================================================
# Hardware  : Raspberry Pi 4B 8GB + Pi Cam V2
# Primary   : MobileNetV2 backbone → steering + speed heads
# Detection : YOLOv11n ONNX (runs every N frames, CPU-optimised)
# Strategy  : BC pre-training (+ augmentation) + Online RL + EMA
# Outputs   : steering ∈ [-1, 1]  (-1=left, +1=right)
#             speed    ∈ [ 0, 1]
# Target FPS: 12–20 fps on Pi 4 CPU
# ============================================================

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import sys
import signal
import copy
from collections import deque

# ── optional ONNX Runtime (install: pip install onnxruntime) ──
try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False
    print("[WARN] onnxruntime not found — PyTorch fallback active")

# ============================================================
# CONFIGURATION
# ============================================================

CFG = {
    # Input
    "img_size":              96,    # 96×96 — faster than 112 on Pi CPU

    # Backbone
    "mobilenet_checkpoint":  "mobilenet_lane.pth",
    "onnx_path":             "mobilenet_lane.onnx",
    "use_onnx":              True,  # flip to False to force PyTorch path

    # YOLO
    "yolo_pt_path":          "yolo11n.pt",   # converted from zip
    "yolo_onnx_path":        "yolo11n.onnx",
    "yolo_input_size":       320,            # 320×320 — fastest YOLO setting
    "yolo_conf_thresh":      0.40,
    "yolo_every_n_frames":   4,              # run YOLO every 4 frames to save CPU
    "obstacle_classes":      {0, 1, 2, 3, 5, 7},  # COCO: person,bicycle,car,moto,bus,truck

    # Training
    "lr":                    3e-4,
    "batch_size":            8,
    "memory_maxlen":         600,
    "train_every_n":         8,

    # EMA
    "ema_decay":             0.995,   # shadow weights used for inference

    # Reward
    "w_lane_center":         1.2,
    "w_speed":               0.3,
    "w_lane_conf":           0.5,
    "w_steer_smooth":        0.2,
    "penalty_lane_lost":     2.0,
    "penalty_collision":     15.0,
    "penalty_boundary":      10.0,

    # Safety
    "lane_conf_stop_thresh": 0.15,
    "max_speed":             0.55,
    "collision_pixel_thresh":0.35,
    "obs_prob_thresh":       0.65,  # YOLO-based override threshold

    "device":                "cpu",
}

device = torch.device(CFG["device"])

# ============================================================
# MOTOR INTERFACE
# ============================================================
# Replace set_controls() body with actual pigpio/RPi.GPIO calls.
#
#   import pigpio
#   pi = pigpio.pi()
#   STEER_PIN, THROTTLE_PIN = 18, 19
#   def _us(v, lo=1000, hi=2000): return int(lo + (v + 1) / 2 * (hi - lo))
#   pi.set_servo_pulsewidth(STEER_PIN,    _us(steering))
#   pi.set_servo_pulsewidth(THROTTLE_PIN, _us(speed))

class MotorController:
    def set_controls(self, steering: float, speed: float):
        # ── INSERT GPIO CODE HERE ──────────────────────────────
        pass

    def stop(self):
        self.set_controls(0.0, 0.0)

motor = MotorController()

# ============================================================
# SAFE-STOP
# ============================================================

_safe_stop_triggered = False

def safe_stop(reason: str = ""):
    global _safe_stop_triggered
    _safe_stop_triggered = True
    motor.stop()
    print(f"\n[SAFE-STOP] {reason}")

signal.signal(signal.SIGINT,  lambda s, f: safe_stop("SIGINT"))
signal.signal(signal.SIGTERM, lambda s, f: safe_stop("SIGTERM"))

# ============================================================
# MODEL — MobileNetV2 backbone + dual head
# ============================================================
# Input  : (B, 3, 96, 96)  normalised [0, 1]
# Outputs: steering ∈ (-1,1),  speed ∈ (0,1)
#
# MobileNetV2 pretrained features (frozen first 7 layers) →
# custom AdaptiveAvgPool → steering head + speed head.
# Params added: ~12k.  Frozen backbone inference: ~30–45ms on Pi 4.

class LaneNet(nn.Module):
    """MobileNetV2 backbone with steering and speed heads."""

    def __init__(self, freeze_layers: int = 7):
        super().__init__()
        from torchvision.models import mobilenet_v2
        base = mobilenet_v2(weights=None)           # no internet on Pi

        # Use MobileNetV2 feature extractor only
        self.backbone = base.features              # outputs (B, 1280, 3, 3) @ 96x96 in
        self.pool     = nn.AdaptiveAvgPool2d(1)    # → (B, 1280)

        # Freeze early layers to keep inference fast and avoid overfitting
        for i, layer in enumerate(self.backbone):
            if i < freeze_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        # Shared neck
        self.neck = nn.Sequential(
            nn.Linear(1280, 128), nn.ReLU6(),
            nn.Dropout(0.2),
        )

        # Steering head: Tanh → (-1, 1)
        self.steer_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32,  1),  nn.Tanh()
        )

        # Speed head: Sigmoid → (0, 1)
        self.speed_head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32,  1),  nn.Sigmoid()
        )

    def forward(self, x):
        feat  = self.pool(self.backbone(x)).flatten(1)
        neck  = self.neck(feat)
        steer = self.steer_head(neck).squeeze(-1)
        speed = self.speed_head(neck).squeeze(-1)
        return steer, speed


model     = LaneNet(freeze_layers=7).to(device)
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=CFG["lr"]
)

# ── EMA shadow weights ─────────────────────────────────────
ema_model  = copy.deepcopy(model)
_ema_decay = CFG["ema_decay"]

def update_ema():
    """Blend live weights into EMA shadow after each gradient step."""
    with torch.no_grad():
        for ema_p, live_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(_ema_decay).add_(live_p.data, alpha=1.0 - _ema_decay)

# ── Load checkpoint if available ──────────────────────────
_CKPT = CFG["mobilenet_checkpoint"]
if os.path.exists(_CKPT):
    state = torch.load(_CKPT, map_location=device)
    model.load_state_dict(state)
    ema_model.load_state_dict(state)
    print(f"[INFO] Loaded checkpoint: {_CKPT}")

# ── ONNX session (populated by export_onnx() or at run start) ──
_ort_session = None

def export_onnx():
    """Export EMA model to ONNX for faster Pi inference."""
    if not ORT_AVAILABLE:
        print("[WARN] onnxruntime not installed — skipping ONNX export")
        return

    path = CFG["onnx_path"]
    dummy = torch.zeros(1, 3, CFG["img_size"], CFG["img_size"])
    ema_model.eval()
    torch.onnx.export(
        ema_model, dummy, path,
        input_names=["input"],
        output_names=["steering", "speed"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=12,
    )
    print(f"[ONNX] Exported to {path}")

    # Verify it loads cleanly
    global _ort_session
    _ort_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print("[ONNX] Session ready — using ONNX Runtime for inference")


def load_onnx_session():
    """Load existing ONNX session at startup."""
    global _ort_session
    if not ORT_AVAILABLE:
        return
    p = CFG["onnx_path"]
    if os.path.exists(p):
        _ort_session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        print(f"[ONNX] Loaded session from {p}")


def infer(tensor: torch.Tensor):
    """
    Run steering+speed inference.
    Uses ONNX Runtime when available (20–35% faster on ARM),
    falls back to PyTorch EMA model.
    """
    if _ort_session is not None and CFG["use_onnx"]:
        np_in = tensor.numpy()
        out   = _ort_session.run(None, {"input": np_in})
        return float(out[0][0]), float(out[1][0])
    else:
        ema_model.eval()
        with torch.no_grad():
            s, sp = ema_model(tensor)
        return s.item(), sp.item()

# ============================================================
# YOLO OBJECT DETECTOR  (runs every N frames)
# ============================================================
# Expects yolo11n.onnx built from the uploaded yolo11n_pt.zip.
# Export once on a desktop:
#   from ultralytics import YOLO
#   YOLO("yolo11n.pt").export(format="onnx", imgsz=320, simplify=True)
# Then copy yolo11n.onnx to the Pi alongside this script.

_yolo_session  = None
_yolo_anchors  = None   # unused for YOLOv8/11 (anchor-free)

def _load_yolo_onnx():
    global _yolo_session
    p = CFG["yolo_onnx_path"]
    if not ORT_AVAILABLE:
        print("[YOLO] onnxruntime not available — YOLO disabled")
        return
    if not os.path.exists(p):
        print(f"[YOLO] {p} not found — YOLO disabled. Export with ultralytics first.")
        return
    _yolo_session = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print(f"[YOLO] Session loaded from {p}")

_load_yolo_onnx()


def _preprocess_yolo(frame: np.ndarray) -> np.ndarray:
    """Letterbox-resize frame to yolo_input_size × yolo_input_size, NCHW float32."""
    sz   = CFG["yolo_input_size"]
    img  = cv2.resize(frame, (sz, sz), interpolation=cv2.INTER_LINEAR)
    img  = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    return np.ascontiguousarray(img.transpose(2, 0, 1))[None]  # (1,3,sz,sz)


def _nms(boxes, scores, iou_thresh=0.45):
    """Simple greedy NMS — avoids cv2.dnn dependency."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas  = (x2 - x1) * (y2 - y1)
    order  = scores.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2 - xx1)
        h   = np.maximum(0, yy2 - yy1)
        iou = (w * h) / (areas[i] + areas[order[1:]] - w * h + 1e-6)
        order = order[1:][iou < iou_thresh]
    return keep


def detect_objects(frame: np.ndarray):
    """
    Run YOLOv11n ONNX on frame.
    Returns (obstacle_detected: bool, closest_box_cx: float | None,
             obs_proximity_score: float).

    obstacle_detected     — True if any obstacle-class object found
    closest_box_cx        — normalised [0,1] centre-x of the largest box
    obs_proximity_score   — area fraction of largest box (0 = none, 1 = full frame)
    """
    if _yolo_session is None:
        return False, None, 0.0

    sz  = CFG["yolo_input_size"]
    inp = _preprocess_yolo(frame)

    # YOLOv8/11 output: (1, 84, 8400) → (cx,cy,w,h, 80 class scores)
    raw   = _yolo_session.run(None, {_yolo_session.get_inputs()[0].name: inp})[0]
    preds = raw[0].T                         # (8400, 84)

    class_scores = preds[:, 4:]              # (8400, 80)
    class_ids    = class_scores.argmax(1)
    confidences  = class_scores.max(1)

    mask = (confidences >= CFG["yolo_conf_thresh"]) & \
           np.isin(class_ids, list(CFG["obstacle_classes"]))

    if not mask.any():
        return False, None, 0.0

    preds_f = preds[mask]
    cx, cy, bw, bh = preds_f[:,0], preds_f[:,1], preds_f[:,2], preds_f[:,3]
    x1 = cx - bw / 2;  y1 = cy - bh / 2
    x2 = cx + bw / 2;  y2 = cy + bh / 2

    boxes  = np.stack([x1, y1, x2, y2], axis=1)
    confs  = confidences[mask]
    keep   = _nms(boxes, confs)

    if not keep:
        return False, None, 0.0

    # Pick the box with the largest area (most proximate threat)
    areas  = (boxes[keep, 2] - boxes[keep, 0]) * (boxes[keep, 3] - boxes[keep, 1])
    best   = keep[areas.argmax()]
    box    = boxes[best]
    area_frac = float(areas.max() / (sz * sz))

    # Normalise centre-x to [0, 1] relative to original frame
    h_orig, w_orig = frame.shape[:2]
    cx_norm = float((box[0] + box[2]) / 2.0 / sz)  # already in [0,1]

    return True, cx_norm, area_frac

# ============================================================
# FALLBACK CONTROL LOGIC
# ============================================================
# Single entry point for steering/speed decisions.
# MobileNetV2 drives normally.
# YOLO triggers an override ONLY when an obstacle is detected:
#   • Slow down proportional to proximity
#   • Steer away from obstacle centre
# No three separate heads — cleaner, faster, easier to tune.

def fallback_control(
    steer_base: float,
    speed_base: float,
    obs_detected: bool,
    obs_cx: float | None,
    obs_proximity: float,
) -> tuple[float, float]:
    """
    Merge MobileNetV2 steering/speed with YOLO obstacle signal.

    Returns (steering, speed) with fallback applied.
    """
    if not obs_detected or obs_cx is None:
        return steer_base, speed_base     # clean path — use model directly

    # ── Speed: slow down proportional to obstacle size ────────
    # obstacle fills 5% → mild brake; 20%+ → near-stop
    proximity_factor = min(obs_proximity / 0.20, 1.0)
    safe_speed = speed_base * (1.0 - 0.85 * proximity_factor)

    # ── Steer: dodge away from obstacle centre ────────────────
    # obs_cx ∈ [0, 1]; 0.5 = centre.  Push steer away.
    dodge = (0.5 - obs_cx) * 2.0           # [-1, 1]: -1 if obs on right, +1 if left
    # Blend: strong dodge when close, gentle when far
    blend_alpha = proximity_factor * 0.7   # max 70% dodge influence
    safe_steer  = (1.0 - blend_alpha) * steer_base + blend_alpha * dodge

    safe_steer = float(np.clip(safe_steer, -1.0, 1.0))
    safe_speed = float(np.clip(safe_speed,  0.0, CFG["max_speed"]))
    return safe_steer, safe_speed

# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(frame: np.ndarray) -> torch.Tensor:
    """
    Bottom-half crop → resize to img_size → CLAHE → normalise → tensor.
    """
    h, w = frame.shape[:2]
    roi  = frame[h // 2:, :]
    roi  = cv2.resize(roi, (CFG["img_size"], CFG["img_size"]))

    # CLAHE on L channel (robust to variable arena lighting)
    lab  = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    roi  = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    return torch.from_numpy(
        roi.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device)

# ============================================================
# LANE ESTIMATION  (classical CV — feeds reward signal)
# ============================================================

def estimate_lane(frame: np.ndarray):
    """Returns (lane_center_x, confidence ∈ [0,1])."""
    h, w = frame.shape[:2]
    roi  = frame[h // 2:, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40,
                             minLineLength=30, maxLineGap=20)
    if lines is None:
        return w / 2, 0.0

    centers = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if 20 < angle < 85:
            centers.append((x1 + x2) / 2)

    if not centers:
        return w / 2, 0.0

    return float(np.median(centers)), min(len(centers) / 8.0, 1.0)

# ============================================================
# OPTICAL-FLOW COLLISION HEURISTIC
# ============================================================

_prev_gray = None

def detect_collision_optical_flow(frame: np.ndarray) -> bool:
    global _prev_gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    collision = False
    if _prev_gray is not None:
        flow = cv2.calcOpticalFlowFarneback(
            _prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        if np.sqrt(flow[...,0]**2 + flow[...,1]**2).mean() < 0.4:
            collision = True
    _prev_gray = gray
    return collision

# ============================================================
# REWARD FUNCTION
# ============================================================

def compute_reward(
    lane_center: float, lane_conf: float,
    steering: float,    speed: float,
    collision: bool,    frame_width: int = 320,
    prev_steering: float = 0.0
) -> float:
    img_center = frame_width / 2
    error      = (lane_center - img_center) / img_center  # [-1, 1]

    reward  =  CFG["w_lane_center"] * (1.0 - abs(error))
    reward +=  CFG["w_speed"]       * min(speed, CFG["max_speed"])
    reward +=  CFG["w_lane_conf"]   * lane_conf
    reward -=  CFG["w_steer_smooth"]* abs(steering - prev_steering)

    if lane_conf < 0.2:
        reward -= CFG["penalty_lane_lost"]
    if collision:
        reward -= CFG["penalty_collision"]

    return float(reward)

# ============================================================
# SAFETY OVERRIDE
# ============================================================

def safe_control(
    steering: float, speed: float,
    lane_conf: float, collision: bool
) -> tuple[float, float]:
    if collision or lane_conf < CFG["lane_conf_stop_thresh"]:
        return 0.0, 0.0
    speed = min(speed, CFG["max_speed"])
    speed *= (1.0 - 0.4 * abs(steering))   # slow on tight turns
    return float(np.clip(steering, -1.0, 1.0)), float(speed)

# ============================================================
# REPLAY MEMORY
# ============================================================

memory          = deque(maxlen=CFG["memory_maxlen"])
_frame_count    = 0
_reward_baseline = 0.0
_BASELINE_ALPHA  = 0.05
_last_ckpt_frame = 0

# ============================================================
# ONLINE TRAINING  (policy gradient + EMA update)
# ============================================================

def maybe_train():
    global _reward_baseline

    n = CFG["batch_size"]
    if len(memory) < n:
        return None

    batch   = list(memory)[-n:]
    states  = torch.cat([b["state"]  for b in batch]).to(device)
    rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32).to(device)

    # Advantage with running baseline
    advantage = rewards - _reward_baseline
    _reward_baseline = (
        (1 - _BASELINE_ALPHA) * _reward_baseline +
        _BASELINE_ALPHA * rewards.mean().item()
    )

    pred_steer, pred_speed = model(states)

    steer_loss = -(advantage * pred_steer).mean()
    speed_loss = -(advantage * pred_speed).mean()
    loss       = steer_loss + speed_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ── EMA update after every gradient step ──────────────────
    update_ema()

    return loss.item()

# ============================================================
# DATA AUGMENTATION  (used during BC training)
# ============================================================

def augment(tensor: torch.Tensor, steering: float) -> tuple[torch.Tensor, float]:
    """
    Apply random augmentations to a single (1,3,H,W) tensor.
    Returns augmented tensor and (possibly mirrored) steering label.

    Augmentations implemented:
    1. Brightness / contrast jitter ±20%
    2. Horizontal flip  (mirrors steering label)
    3. Additive Gaussian noise  σ=0.02
    """
    img = tensor.clone().squeeze(0)  # (3, H, W) in [0, 1]

    # 1. Brightness jitter
    if np.random.rand() < 0.5:
        factor = 1.0 + np.random.uniform(-0.2, 0.2)
        img    = (img * factor).clamp(0, 1)

    # 2. Contrast jitter
    if np.random.rand() < 0.5:
        mean   = img.mean()
        factor = 1.0 + np.random.uniform(-0.2, 0.2)
        img    = ((img - mean) * factor + mean).clamp(0, 1)

    # 3. Horizontal flip — must mirror steering
    if np.random.rand() < 0.5:
        img      = torch.flip(img, dims=[2])
        steering = -steering

    # 4. Gaussian noise
    if np.random.rand() < 0.5:
        noise = torch.randn_like(img) * 0.02
        img   = (img + noise).clamp(0, 1)

    return img.unsqueeze(0), float(steering)

# ============================================================
# BEHAVIOURAL CLONING  (--collect / --train_bc)
# ============================================================

def collect_bc_data(cap):
    """Manual drive mode: WASD to steer, Q to quit. Saves bc_data.pth."""
    print("[BC] Manual drive mode. WASD = steer/speed, Q = quit.")
    bc_data  = []
    steering = 0.0
    speed    = 0.2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('a'): steering = max(steering - 0.1, -1.0)
        elif key == ord('d'): steering = min(steering + 0.1,  1.0)
        elif key == ord('w'): speed    = min(speed    + 0.05,  1.0)
        elif key == ord('s'): speed    = max(speed    - 0.05,  0.0)
        elif key == ord('q'): break
        else:                 steering *= 0.85   # auto-centre

        tensor = preprocess(frame)
        bc_data.append({"state": tensor.cpu(), "steering": steering, "speed": speed})
        motor.set_controls(steering, speed)

        cv2.putText(frame, f"BC | S:{steering:+.2f} V:{speed:.2f}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("TRAFFIQ — BC Collect", frame)

    torch.save(bc_data, "bc_data.pth")
    print(f"[BC] Saved {len(bc_data)} samples → bc_data.pth")
    motor.stop()


def train_bc():
    """
    Supervised training on collected BC data with data augmentation.
    Each real sample is augmented on-the-fly → effectively 3–4× dataset.
    After training, exports EMA weights to ONNX.
    """
    if not os.path.exists("bc_data.pth"):
        print("[BC] bc_data.pth not found — run --collect first.")
        return

    data = torch.load("bc_data.pth")
    print(f"[BC] Training on {len(data)} samples (+ online augmentation)...")

    bc_opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )
    mse = nn.MSELoss()

    for epoch in range(25):
        np.random.shuffle(data)
        total_loss = 0.0

        for i in range(0, len(data) - CFG["batch_size"], CFG["batch_size"]):
            batch   = data[i : i + CFG["batch_size"]]

            # Build augmented batch
            aug_states, aug_steers, aug_speeds = [], [], []
            for s in batch:
                t, st = augment(s["state"], s["steering"])
                aug_states.append(t)
                aug_steers.append(st)
                aug_speeds.append(s["speed"])

            states  = torch.cat(aug_states).to(device)
            t_steer = torch.tensor(aug_steers, dtype=torch.float32).to(device)
            t_speed = torch.tensor(aug_speeds, dtype=torch.float32).to(device)

            p_steer, p_speed = model(states)
            loss = mse(p_steer, t_steer) + mse(p_speed, t_speed)

            bc_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            bc_opt.step()

            # EMA update every step
            update_ema()
            total_loss += loss.item()

        print(f"  Epoch {epoch+1:02d} | loss {total_loss:.4f}")

    torch.save(model.state_dict(), _CKPT)
    print(f"[BC] Saved model to {_CKPT}")

    # Export EMA weights to ONNX for faster Pi inference
    export_onnx()

# ============================================================
# CHECKPOINT AUTO-SAVE  (every 200 frames)
# ============================================================

def maybe_save_checkpoint(frame_count: int):
    global _last_ckpt_frame
    if frame_count - _last_ckpt_frame >= 200:
        torch.save(model.state_dict(), _CKPT)
        torch.save(ema_model.state_dict(), _CKPT.replace(".pth", "_ema.pth"))
        _last_ckpt_frame = frame_count

# ============================================================
# MAIN LOOP
# ============================================================

def run(cap):
    global _frame_count

    # Load ONNX session if available
    load_onnx_session()

    print("=" * 55)
    print("  TRAFFIQ Agent v2 — Autonomous Run")
    print(f"  Inference: {'ONNX Runtime' if _ort_session else 'PyTorch EMA'}")
    print(f"  YOLO:      {'enabled' if _yolo_session else 'disabled'}")
    print("  Press Q to quit cleanly")
    print("=" * 55)

    ema_model.eval()
    prev_steering  = 0.0
    frame_w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    _yolo_frame_n  = CFG["yolo_every_n_frames"]

    # Cache last YOLO result (reused between YOLO frames)
    last_obs_detected  = False
    last_obs_cx        = None
    last_obs_proximity = 0.0

    while not _safe_stop_triggered:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            safe_stop("Camera feed lost")
            break

        # ── MobileNetV2 Inference ─────────────────────────────
        try:
            input_tensor    = preprocess(frame)
            steer_base, speed_base = infer(input_tensor)
        except Exception as e:
            safe_stop(f"Model crash: {e}")
            break

        # ── YOLO Object Detection (every N frames) ─────────────
        if _yolo_session and (_frame_count % _yolo_frame_n == 0):
            try:
                last_obs_detected, last_obs_cx, last_obs_proximity = \
                    detect_objects(frame)
            except Exception:
                pass   # YOLO failure is non-fatal

        # ── Fallback Logic: YOLO overrides steering/speed ─────
        steering, speed = fallback_control(
            steer_base, speed_base,
            last_obs_detected, last_obs_cx, last_obs_proximity
        )

        # ── Classical Perception ──────────────────────────────
        lane_center, lane_conf = estimate_lane(frame)
        collision_flow         = detect_collision_optical_flow(frame)
        collision = collision_flow or (last_obs_proximity > 0.20)

        # ── Safety Override ───────────────────────────────────
        ctrl_steer, ctrl_speed = safe_control(steering, speed, lane_conf, collision)

        # ── Motor Output ──────────────────────────────────────
        motor.set_controls(ctrl_steer, ctrl_speed)

        # ── Reward & Memory ───────────────────────────────────
        reward = compute_reward(
            lane_center, lane_conf,
            ctrl_steer, ctrl_speed, collision,
            frame_width=frame_w, prev_steering=prev_steering
        )
        memory.append({
            "state":   input_tensor,
            "steering": ctrl_steer,
            "speed":    ctrl_speed,
            "reward":   reward,
            "collision": collision,
        })
        prev_steering  = ctrl_steer
        _frame_count  += 1

        # ── Online RL update ──────────────────────────────────
        loss_val = None
        if _frame_count % CFG["train_every_n"] == 0:
            model.train()
            loss_val = maybe_train()
            model.eval()

        # ── Periodic checkpoint ───────────────────────────────
        maybe_save_checkpoint(_frame_count)

        # ── HUD ───────────────────────────────────────────────
        elapsed_ms = (time.time() - t0) * 1000
        col        = (0, 0, 255) if collision else (0, 255, 0)
        obs_tag    = f"OBS {last_obs_proximity:.2f}" if last_obs_detected else "clear"
        hud_lines  = [
            f"Steer : {ctrl_steer:+.3f}",
            f"Speed : {ctrl_speed:.3f}",
            f"Lane  : {lane_conf:.2f}",
            f"Detect: {obs_tag}",
            f"Rwrd  : {reward:+.2f}",
            f"Infer : {elapsed_ms:.0f}ms",
        ]
        if loss_val is not None:
            hud_lines.append(f"Loss  : {loss_val:.4f}")
        if collision:
            hud_lines.append("!! COLLISION !!")

        for i, txt in enumerate(hud_lines):
            cv2.putText(frame, txt, (10, 26 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)

        cv2.imshow("TRAFFIQ — Agent v2", frame)

        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0
        print(
            f"[{_frame_count:05d}] "
            f"steer={ctrl_steer:+.3f}  spd={ctrl_speed:.3f}  "
            f"conf={lane_conf:.2f}  obs={obs_tag}  "
            f"rwrd={reward:+.2f}  {fps:.1f}fps"
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            safe_stop("User quit")
            break

    motor.stop()
    cap.release()
    cv2.destroyAllWindows()
    torch.save(model.state_dict(), _CKPT)
    torch.save(ema_model.state_dict(), _CKPT.replace(".pth", "_ema.pth"))
    print("[INFO] Checkpoints saved.")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "--run"

    if mode == "--export_onnx":
        export_onnx()
        sys.exit(0)

    cap = cv2.VideoCapture(0)   # change to "video.mp4" for offline testing
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        sys.exit(1)

    if mode == "--collect":
        collect_bc_data(cap)
    elif mode == "--train_bc":
        cap.release()
        train_bc()
    elif mode == "--run":
        run(cap)
    else:
        print(
            "Usage: python traffiq_agent.py "
            "[--run | --collect | --train_bc | --export_onnx]"
        )

    cap.release()
    cv2.destroyAllWindows()
