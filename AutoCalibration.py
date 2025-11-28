#!/usr/bin/env python3
import os
import cv2
import numpy as np
from glob import glob
import sys
import time

# ============================================================
# CONFIGURATION
# ============================================================

BASE = "/Users/acalapai/Desktop/Collage/calib_videos"

# Checkerboard: 14×10 tiles → 13×9 inner corners
CHESSBOARD = (13, 9)
SQUARE_SIZE_M = 0.025  # 25 mm

MAX_FRAMES_PER_CAMERA = 80
FRAME_STEP = 20

OUT_DIR = os.path.join(BASE, "calibration_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Helpers
# ============================================================

def extract_cam_id(path):
    name = os.path.basename(path)
    for p in ["102", "108", "113", "117"]:
        if f"_{p}_" in name or name.startswith(p+"_"):
            return p
    return "XXX"

def bar(prefix, i, total):
    width = 30
    ratio = min(1.0, i/float(total))
    fill = int(ratio * width)
    line = "#" * fill + "-" * (width-fill)
    sys.stdout.write(f"\r{prefix} [{line}] {i}/{total}")
    sys.stdout.flush()

# ============================================================
# Load videos
# ============================================================

VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if not VIDEOS:
    raise RuntimeError("No videos found")

print("\n=== Found videos ===")
for v in VIDEOS:
    print(" -", v)

# Checkerboard 3D points
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

per_cam_objpoints = {}
per_cam_imgpoints = {}
per_cam_imgsize = {}

# ============================================================
# STEP 1 — Detect chessboard in each camera (with progress bar)
# ============================================================

print("\n=== STEP 1: Detecting checkerboards ===")

for video_path in VIDEOS:

    cam_id = extract_cam_id(video_path)
    print(f"\nCamera {cam_id}: {os.path.basename(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("  ERROR opening video.")
        continue

    per_cam_objpoints[cam_id] = []
    per_cam_imgpoints[cam_id] = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Total frames: {total}")

    used = 0
    idx = 0

    print("  Searching frames:")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # draw progress bar based on frame index
        bar("    Progress", idx, total)

        if idx % FRAME_STEP == 0 and used < MAX_FRAMES_PER_CAMERA:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_cb, corners = cv2.findChessboardCorners(gray, CHESSBOARD)

            if ret_cb:
                per_cam_objpoints[cam_id].append(objp)
                per_cam_imgpoints[cam_id].append(corners)
                per_cam_imgsize[cam_id] = gray.shape[::-1]
                used += 1

                print(f"\n    ✓ Found corners at frame {idx} "
                      f"({used}/{MAX_FRAMES_PER_CAMERA})")

        idx += 1
        if used >= MAX_FRAMES_PER_CAMERA:
            break

    cap.release()
    print(f"\n  -> Collected {used} checkerboard frames")

# ============================================================
# STEP 2 — Intrinsic calibration
# ============================================================

print("\n=== STEP 2: Calibrating intrinsics ===")

intrinsics = {}
extrinsics = {}

for cam_id in per_cam_objpoints.keys():

    objpoints = per_cam_objpoints[cam_id]
    imgpoints = per_cam_imgpoints[cam_id]

    if len(objpoints) < 10:
        print(f"  WARNING: Not enough frames for {cam_id}, skipping.")
        continue

    print(f"\nCalibrating camera {cam_id} ({len(objpoints)} frames)")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        per_cam_imgsize[cam_id],
        None,
        None,
        flags=cv2.CALIB_FIX_K3
    )

    print(f"  RMS: {rms:.4f}")
    intrinsics[cam_id] = (K, dist, rms)

# ============================================================
# STEP 3 — Multi-camera extrinsics (relative to REF)
# ============================================================

REF = sorted(intrinsics.keys())[0]
print(f"\n=== STEP 3: Solving extrinsics (REF = {REF}) ===")

for cam_id in sorted(intrinsics.keys()):
    if cam_id == REF:
        extrinsics[cam_id] = (np.eye(3), np.zeros((3,1)))
        print(f"  {cam_id}: reference camera")
        continue

    print(f"\nSolving extrinsics for camera {cam_id}")

    # just use first detection match
    K, dist, _ = intrinsics[cam_id]
    imgp = per_cam_imgpoints[cam_id][0]
    objp_single = objp

    ok, rvec, tvec = cv2.solvePnP(
        objp_single, imgp, K, dist, flags=cv2.SOLVEPNP_ITERATIVE
    )

    R, _ = cv2.Rodrigues(rvec)
    extrinsics[cam_id] = (R, tvec)

    print(f"  R =\n{R}")
    print(f"  t = {tvec.ravel()}")

# ============================================================
# STEP 4 — Save all parameters
# ============================================================

print("\n=== STEP 4: Saving ===")

np.save(os.path.join(OUT_DIR, "intrinsics.npy"), intrinsics)
np.save(os.path.join(OUT_DIR, "extrinsics.npy"), extrinsics)

print("\n✔ Calibration completed")
print(f"Saved to: {OUT_DIR}")