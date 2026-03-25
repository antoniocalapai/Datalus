#!/usr/bin/env python3
"""
Datalus Calibration — Step 2: Intrinsic Calibration
Detects checkerboard corners in extracted frames and runs cv2.calibrateCamera
per camera. Saves intrinsics as intrinsics.npz in the output directory.
Run after DatulusCalib_Step1_Frames.py (Step 1).
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FRAMES_ROOT   = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration/frames"
OUTPUT_DIR    = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"

CHESSBOARD    = (13, 9)    # inner corners (width, height)
SQUARE_SIZE_M = 40.0 / 1000.0  # 40 mm in meters

SENSOR_WIDTH_MM = 11.2
FOCAL_LENGTH_MM = 8.0

CAMERA_IDS    = ["102", "108", "113", "117"]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def bar(prefix, i, total, found=0):
    width = 30
    ratio = min(1.0, i / max(total, 1))
    fill  = int(ratio * width)
    line  = "#" * fill + "-" * (width - fill)
    sys.stdout.write(f"\r  {prefix} [{line}] {i}/{total}  found: {found}")
    sys.stdout.flush()

# ─── PARALLEL DETECTION ───────────────────────────────────────────────────────

def _detect_one(args):
    """Worker: detect and refine checkerboard corners in a single frame."""
    img_path, chessboard = args
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    found, corners = cv2.findChessboardCorners(
        gray, chessboard,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    )
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
        )
        return corners
    return None


# ─── STEP 2: Intrinsic Calibration ────────────────────────────────────────────

def calibrate_intrinsics(frames_root, camera_ids, chessboard, square_size_m,
                          focal_mm, sensor_w_mm):
    """
    For each camera:
      - detect checkerboard corners in all extracted frames
      - run cv2.calibrateCamera with a spec-derived focal length as initial guess
    Returns:
      intrinsics: dict  cam_id -> {"K": 3x3, "dist": 1x5, "rms": float,
                                   "n_frames": int, "image_size": (w,h)}
    """
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= square_size_m

    intrinsics = {}

    for cam_id in camera_ids:
        cam_dir = Path(frames_root) / cam_id
        images  = sorted(cam_dir.glob("*.png"))

        if not images:
            print(f"\n  [WARN] No frames found for cam {cam_id} in {cam_dir}")
            continue

        # read one image to get resolution
        sample = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        if sample is None:
            print(f"\n  [ERROR] Cannot read frame: {images[0]}")
            sys.exit(1)
        h, w = sample.shape
        f_px = focal_mm * w / sensor_w_mm

        print(f"\n  cam {cam_id}: {w}x{h} | prior focal = {f_px:.1f} px | "
              f"{len(images)} frames to scan")

        obj_pts, img_pts = [], []
        total   = len(images)
        n_cores = max(1, int(mp.cpu_count() * 0.75))
        args    = [(str(p), chessboard) for p in images]

        print(f"    using {n_cores} CPU cores")

        completed = 0
        with mp.Pool(processes=n_cores) as pool:
            for result in pool.imap(_detect_one, args):
                completed += 1
                if result is not None:
                    obj_pts.append(objp)
                    img_pts.append(result)
                bar("detecting", completed, total, found=len(obj_pts))

        print(f"\n    detections: {len(obj_pts)} / {total}")

        if len(obj_pts) < 3:
            print(f"    [WARN] Too few detections for cam {cam_id} ({len(obj_pts)}) — skipping.")
            continue

        K_init = np.array([
            [f_px, 0,    w / 2.0],
            [0,    f_px, h / 2.0],
            [0,    0,    1.0    ]
        ], dtype=np.float64)

        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, (w, h),
            K_init.copy(), None,
            flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
        )

        # pad distortion to 5 coefficients [k1, k2, p1, p2, k3]
        d = np.zeros(5)
        d[:min(5, dist.ravel().size)] = dist.ravel()[:5]

        intrinsics[cam_id] = {
            "K":          K,
            "dist":       d.reshape(1, 5),
            "rms":        rms,
            "n_frames":   len(obj_pts),
            "image_size": (w, h),
        }

        print(f"    RMS reprojection error: {rms:.4f} px")
        print(f"    fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
              f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
        print(f"    dist: {d}")

    return intrinsics


def save_intrinsics(intrinsics, output_dir):
    """Save all camera intrinsics to a single .npz file."""
    out_path = Path(output_dir) / "intrinsics.npz"
    save_dict = {}
    for cam_id, data in intrinsics.items():
        save_dict[f"{cam_id}_K"]          = data["K"]
        save_dict[f"{cam_id}_dist"]       = data["dist"]
        save_dict[f"{cam_id}_rms"]        = np.array([data["rms"]])
        save_dict[f"{cam_id}_image_size"] = np.array(data["image_size"])
    np.savez(str(out_path), **save_dict)
    print(f"\n  Intrinsics saved to {out_path}")
    return out_path


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DATALUS — STEP 2: INTRINSIC CALIBRATION")
    print("=" * 60)

    intrinsics = calibrate_intrinsics(
        FRAMES_ROOT, CAMERA_IDS, CHESSBOARD, SQUARE_SIZE_M,
        FOCAL_LENGTH_MM, SENSOR_WIDTH_MM
    )

    if not intrinsics:
        print("\n[ERROR] No cameras calibrated. Check FRAMES_ROOT and checkerboard config.")
        sys.exit(1)

    save_intrinsics(intrinsics, OUTPUT_DIR)

    print("\nSummary:")
    for cam_id, data in intrinsics.items():
        print(f"  cam {cam_id}: RMS={data['rms']:.4f} px  "
              f"({data['n_frames']} frames used)")

    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()