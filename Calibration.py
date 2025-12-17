#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np
from glob import glob

# ============================================================
# CONFIG
# ============================================================
BASE = "/Users/acalapai/Desktop/Collage/calib_videos/"
OUT_DIR = os.path.join(BASE, "jarvis_pairs_out")
os.makedirs(OUT_DIR, exist_ok=True)

# Inner corners (columns, rows): e.g., 14×10 squares -> 13×9 inner corners
CHESSBOARD = (13, 9)

# Checker size in meters (25 mm)
SQUARE_SIZE_M = 0.04

# Sampling
FRAME_STEP = 20
MAX_FRAMES_PER_CAMERA = 200
MIN_COMMON_DETECTIONS = 15

# Define the pairs you want to produce
PAIRS = [("102", "113"), ("108", "117"), ("108", "117")]  # edit

# If you want strict camera-ID parsing, list your IDs here:
KNOWN_CAM_IDS = ["102", "108", "113", "117"]

# ============================================================
# Helpers
# ============================================================
def die(msg: str, code: int = 1):
    print(f"\n[ERROR] {msg}\n")
    sys.exit(code)

def bar(prefix, i, total):
    width = 28
    total = max(total, 1)
    ratio = min(1.0, i / float(total))
    fill = int(ratio * width)
    line = "#" * fill + "-" * (width - fill)
    sys.stdout.write(f"\r{prefix} [{line}] {i}/{total}")
    sys.stdout.flush()

def extract_cam_id(path: str) -> str:
    name = os.path.basename(path)
    for p in KNOWN_CAM_IDS:
        # matches: ..._113_... or startswith("113_") or endswith("_113.mp4")
        if f"_{p}_" in name or name.startswith(p + "_") or name.endswith(f"_{p}.mp4"):
            return p
    return "XXX"

def write_stereo_yaml(out_path, K1, d1, K2, d2, R, T, E, F):
    fs = cv2.FileStorage(out_path, cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix1", K1)
    fs.write("distCoeffs1", d1)
    fs.write("cameraMatrix2", K2)
    fs.write("distCoeffs2", d2)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.release()

# ============================================================
# Build object points once
# ============================================================
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

# ============================================================
# Load videos
# ============================================================
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if not VIDEOS:
    die(f"No .mp4 videos found in {BASE}")

print("\n=== Found videos ===")
vid_by_cam = {}
for v in VIDEOS:
    cam = extract_cam_id(v)
    print(f" - {os.path.basename(v)}   -> cam {cam}")
    if cam == "XXX":
        continue
    # keep the first video per cam unless you want multi-session handling
    vid_by_cam.setdefault(cam, v)

missing = [c for c in set(sum(([a, b] for a, b in PAIRS), [])) if c not in vid_by_cam]
if missing:
    die(f"Missing videos for cameras: {missing}. Check filenames/cam IDs.", 2)

# ============================================================
# STEP 1 — detect checkerboards per camera (store frame index + corners)
# ============================================================
print("\n=== STEP 1: Detecting checkerboards per camera ===")

per_cam_objpoints = {}
per_cam_imgpoints = {}
per_cam_frameidx = {}
per_cam_imgsize = {}

for cam_id, video_path in sorted(vid_by_cam.items()):
    print(f"\nCamera {cam_id}: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        die(f"Could not open video for camera {cam_id}: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total}")
    used = 0
    idx = 0

    per_cam_objpoints[cam_id] = []
    per_cam_imgpoints[cam_id] = []
    per_cam_frameidx[cam_id] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if idx % 200 == 0:
            bar("  Progress", idx, total)

        if idx % FRAME_STEP == 0 and used < MAX_FRAMES_PER_CAMERA:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, CHESSBOARD)

            if found:
                # refine corners for better calibration
                corners = cv2.cornerSubPix(
                    gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
                )
                per_cam_objpoints[cam_id].append(objp)
                per_cam_imgpoints[cam_id].append(corners)
                per_cam_frameidx[cam_id].append(idx)
                per_cam_imgsize[cam_id] = gray.shape[::-1]
                used += 1

        idx += 1
        if used >= MAX_FRAMES_PER_CAMERA:
            break

    cap.release()
    print(f"\n  -> detections kept: {used}")

# ============================================================
# STEP 2 — intrinsics per camera (needed because we fix intrinsics in stereoCalibrate)
# ============================================================
print("\n=== STEP 2: Intrinsic calibration per camera ===")

intrinsics = {}  # cam_id -> (K, dist, rms)

for cam_id in sorted(per_cam_objpoints.keys()):
    objpoints = per_cam_objpoints[cam_id]
    imgpoints = per_cam_imgpoints[cam_id]

    if len(objpoints) < 10:
        die(f"Not enough detections for intrinsics on camera {cam_id} (have {len(objpoints)}).", 3)

    rms, K, dist, *_ = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        per_cam_imgsize[cam_id],
        None,
        None,
        flags=cv2.CALIB_FIX_K3
    )
    intrinsics[cam_id] = (K, dist, rms)
    print(f"  cam {cam_id}: RMS={rms:.4f}")

# ============================================================
# STEP 3 — stereo pairs: match common frame indices and write YAML
# ============================================================
print("\n=== STEP 3: Stereo calibration per pair (write JARVIS/OpenCV YAML) ===")

for a, b in PAIRS:
    if a not in intrinsics or b not in intrinsics:
        print(f"  [SKIP] Pair {a}-{b}: missing intrinsics")
        continue

    fa = per_cam_frameidx.get(a, [])
    fb = per_cam_frameidx.get(b, [])
    common = sorted(set(fa).intersection(set(fb)))

    if len(common) < MIN_COMMON_DETECTIONS:
        die(f"Pair {a}-{b}: only {len(common)} common detections. "
            f"(Need >= {MIN_COMMON_DETECTIONS}). Increase MAX/FRAME_STEP or re-record.", 4)

    # frame_idx -> index in imgpoints list
    ia = {f: i for i, f in enumerate(fa)}
    ib = {f: i for i, f in enumerate(fb)}

    objpoints = []
    imgpoints_a = []
    imgpoints_b = []

    # use up to the smallest count
    for f in common[:min(len(common), 200)]:
        objpoints.append(objp)
        imgpoints_a.append(per_cam_imgpoints[a][ia[f]])
        imgpoints_b.append(per_cam_imgpoints[b][ib[f]])

    K1, d1, _ = intrinsics[a]
    K2, d2, _ = intrinsics[b]

    # use camera A size as reference; in practice both must match for good results
    image_size = per_cam_imgsize[a]
    if per_cam_imgsize[a] != per_cam_imgsize[b]:
        die(f"Pair {a}-{b}: image sizes differ: {per_cam_imgsize[a]} vs {per_cam_imgsize[b]}.", 5)

    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9)

    rms, K1o, d1o, K2o, d2o, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_a, imgpoints_b,
        K1, d1, K2, d2,
        image_size,
        criteria=criteria,
        flags=flags
    )

    out_yaml = os.path.join(OUT_DIR, f"stereo_calibration_{a}_{b}.yaml")
    write_stereo_yaml(out_yaml, K1, d1, K2, d2, R, T, E, F)

    print(f"  ✔ Pair {a}-{b}: stereo RMS={rms:.4f} -> {out_yaml}")

print(f"\nDone. YAMLs saved in: {OUT_DIR}\n")