#!/usr/bin/env python3
"""
Quick 3D reconstruction of the PriCaB room from 4 videos using COLMAP (sparse SfM).

Steps:
1. Extract a limited number of frames from each MP4 (per camera).
2. Run COLMAP:
   - feature_extractor
   - exhaustive_matcher
   - mapper
   - model_converter (to TXT)
3. Load points3D.txt + images.txt
4. Show interactive 3D scatter of points + camera centers.
"""

import os
import re
import subprocess
from glob import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

# ================================================================
# CONFIG
# ================================================================
# Folder with your 4 MP4 videos
BASE = "/Users/acalapai/Desktop/Collage"

# Use colmap from PATH
COLMAP = "colmap"

# Frame extraction
MAX_FRAMES_PER_CAM = 60   # how many frames per camera at most
MIN_STEP = 10             # minimum frame step (to avoid using every frame)

# Output structure
SFM_ROOT = os.path.join(BASE, "SfM")
FRAMES_DIR = os.path.join(SFM_ROOT, "frames")       # where extracted images go
DB_PATH = os.path.join(SFM_ROOT, "database.db")     # COLMAP database
SPARSE_DIR = os.path.join(SFM_ROOT, "sparse")       # mapper output (models)

os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)

# ================================================================
# HELPERS
# ================================================================
def extract_cam_id(path: str) -> str:
    """Extract camera ID from filename using __XYZ_ pattern (e.g. __102_)."""
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else "CAM"


def run_cmd(cmd, desc: str):
    """Run a shell command and print a short description."""
    print(f"\n[RUN] {desc}")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] {desc}")


def quaternion_to_rotation(qw, qx, qy, qz):
    """Convert COLMAP quaternion (qw, qx, qy, qz) to 3x3 rotation matrix."""
    q = np.array([qw, qx, qy, qz], dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])
    return R

# ================================================================
# 1. COLLECT VIDEOS
# ================================================================
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if len(VIDEOS) == 0:
    raise RuntimeError(f"No MP4 videos found in {BASE}")

print("Using videos:")
for v in VIDEOS:
    print(" -", v)

# ================================================================
# 2. EXTRACT FRAMES (few per camera)
# ================================================================
print("\n=== Step 1: Extracting frames per camera ===")
for vid in VIDEOS:
    cam_id = extract_cam_id(vid)
    cam_dir = os.path.join(FRAMES_DIR, f"cam_{cam_id}")
    os.makedirs(cam_dir, exist_ok=True)

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        print(f"[WARNING] Could not open video {vid}")
        continue

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"[WARNING] No frames in {vid}")
        cap.release()
        continue

    # decide stride to extract at most MAX_FRAMES_PER_CAM
    step = max(MIN_STEP, total // MAX_FRAMES_PER_CAM)
    print(f"\nVideo {os.path.basename(vid)} (cam {cam_id}): total={total}, step={step}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            out_name = f"{cam_id}_{frame_idx:06d}.png"
            out_path = os.path.join(cam_dir, out_name)
            cv2.imwrite(out_path, frame)
            saved += 1
            if saved >= MAX_FRAMES_PER_CAM:
                break
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frames to {cam_dir}")

# ================================================================
# 3. RUN COLMAP (feature extraction, matching, mapping)
# ================================================================
print("\n=== Step 2: Running COLMAP SfM ===")

# Remove old database if exists
if os.path.exists(DB_PATH):
    os.remove(DB_PATH)
    print(f"Removed old database: {DB_PATH}")

# 3.1 Feature extraction
run_cmd(
    [
        COLMAP, "feature_extractor",
        "--database_path", DB_PATH,
        "--image_path", FRAMES_DIR,
        "--ImageReader.single_camera", "0",
        "--SiftExtraction.peak_threshold", "0.01",
    ],
    "COLMAP feature_extractor",
)

# 3.2 Exhaustive matcher (OK since we have few images)
run_cmd(
    [
        COLMAP, "exhaustive_matcher",
        "--database_path", DB_PATH,
    ],
    "COLMAP exhaustive_matcher",
)

# 3.3 Sparse reconstruction (mapper)
run_cmd(
    [
        COLMAP, "mapper",
        "--database_path", DB_PATH,
        "--image_path", FRAMES_DIR,
        "--output_path", SPARSE_DIR,
    ],
    "COLMAP mapper",
)

# The mapper creates subfolders 0, 1, ... ; we take model 0 by default
MODEL_DIR = os.path.join(SPARSE_DIR, "0")
if not os.path.isdir(MODEL_DIR):
    raise RuntimeError(f"COLMAP did not create model directory {MODEL_DIR}")
print(f"\nUsing sparse model in: {MODEL_DIR}")

# 3.4 Convert model to TXT (points3D.txt, cameras.txt, images.txt)
run_cmd(
    [
        COLMAP, "model_converter",
        "--input_path", MODEL_DIR,
        "--output_path", MODEL_DIR,
        "--output_type", "TXT",
    ],
    "COLMAP model_converter (TXT)",
)

# ================================================================
# 4. LOAD COLMAP OUTPUT (points + camera poses)
# ================================================================
print("\n=== Step 3: Loading COLMAP model ===")

points_path = os.path.join(MODEL_DIR, "points3D.txt")
images_path = os.path.join(MODEL_DIR, "images.txt")

if not os.path.isfile(points_path):
    raise RuntimeError(f"points3D.txt not found in {MODEL_DIR}")
if not os.path.isfile(images_path):
    raise RuntimeError(f"images.txt not found in {MODEL_DIR}")

# 4.1 Load 3D points
pts = []
colors = []
with open(points_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # id, X, Y, Z, R, G, B, error, track...
        X, Y, Z = map(float, parts[1:4])
        R, G, B = map(float, parts[4:7])
        pts.append([X, Y, Z])
        colors.append([R / 255.0, G / 255.0, B / 255.0])

pts = np.array(pts, dtype=float)
colors = np.array(colors, dtype=float)
print(f"Loaded {pts.shape[0]} 3D points")

# 4.2 Load camera centers from images.txt
cam_centers = []
cam_names = []
with open(images_path, "r") as f:
    lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

# Each image uses 2 lines: pose line + 2D measurements line
for i in range(0, len(lines), 2):
    fields = lines[i].split()
    if len(fields) < 10:
        continue
    # id, qw, qx, qy, qz, tx, ty, tz, cam_id, name
    qw, qx, qy, qz = map(float, fields[1:5])
    tx, ty, tz = map(float, fields[5:8])
    name = fields[9]

    R = quaternion_to_rotation(qw, qx, qy, qz)
    t = np.array([[tx], [ty], [tz]])

    # camera center in world coords: C = -R^T * t
    C = -R.T @ t
    cam_centers.append(C.ravel())
    cam_names.append(name)

cam_centers = np.array(cam_centers)
print(f"Loaded {cam_centers.shape[0]} camera poses")

# ================================================================
# 5. VISUALIZE
# ================================================================
print("\n=== Step 4: Showing interactive 3D plot ===")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D points
ax.scatter(
    pts[:, 0],
    pts[:, 1],
    pts[:, 2],
    s=1,
    c=colors,
    alpha=0.7,
    label="3D points",
)

# Camera centers + path
ax.plot(
    cam_centers[:, 0],
    cam_centers[:, 1],
    cam_centers[:, 2],
    "-o",
    markersize=4,
    label="camera centers",
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Sparse 3D reconstruction + camera poses (COLMAP)")
ax.legend()

# Equal-ish aspect
max_range = (pts.max(axis=0) - pts.min(axis=0)).max()
mid = pts.mean(axis=0)
for setter, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
    setter(m - max_range / 2, m + max_range / 2)

plt.tight_layout()
plt.show()

print("\nDONE. You can rotate/zoom the 3D view to inspect the room.")
print(f"All COLMAP artifacts are in: {SFM_ROOT}")