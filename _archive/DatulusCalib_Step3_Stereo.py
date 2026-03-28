#!/usr/bin/env python3
"""
Datalus Calibration — Step 3: Multi-Camera Stereo Calibration

Replaces COLMAP-based Step 3.  Uses synchronized checkerboard detections
already present in frames/ to compute relative camera poses via
cv2.stereoCalibrate — no SIFT, no external dependencies.

How it works
------------
1. Re-detect checkerboard corners for each camera (parallel, keyed by
   (session, frame_idx) so synchronized frames can be matched).
2. For each non-reference camera, find frames where both it and the
   reference camera (REFERENCE_CAM) detected the board simultaneously,
   then run cv2.stereoCalibrate with CALIB_FIX_INTRINSIC.
3. If a camera has too few shared frames with the reference, fall back
   to chaining through any already-solved camera.
4. Triangulate a set of checkerboard corners visible in all cameras to
   produce candidate reference points for world_registration.csv.
5. Save poses to colmap_poses.npz (same format as before, compatible
   with Step 4).

Run after DatulusCalib_Step2_Intrinsics.py.
"""

import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path
import multiprocessing as mp

# ─── LOGGING ──────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()
    def fileno(self): return self.streams[0].fileno()

def _setup_log(log_path):
    from datetime import datetime
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    _f = open(log_path, "a")
    _f.write(f"\n{'='*40}\n{datetime.now():%Y-%m-%d %H:%M:%S}  {Path(sys.argv[0]).name}\n{'='*40}\n")
    sys.stdout = _Tee(sys.__stdout__, _f)
    sys.stderr = _Tee(sys.__stderr__, _f)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FRAMES_ROOT    = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/frames"
INTRINSICS_NPZ = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/intrinsics.npz"
OUTPUT_DIR     = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration"

CHESSBOARD     = (13, 9)
SQUARE_SIZE_MM = 40.0

CAMERA_IDS     = ["102", "108", "113", "117"]
REFERENCE_CAM  = "108"    # all poses expressed relative to this camera
MIN_STEREO_PAIRS = 10     # minimum shared detections required per camera pair

# ─── DETECTION ────────────────────────────────────────────────────────────────

def _detect_corners_keyed(args):
    """
    Worker: detect checkerboard corners in one frame.
    Returns ((session, frame_idx), corners) or None.
    """
    img_path, chessboard = args
    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    found, corners = cv2.findChessboardCorners(
        gray, chessboard,
        flags=(cv2.CALIB_CB_ADAPTIVE_THRESH
               | cv2.CALIB_CB_NORMALIZE_IMAGE
               | cv2.CALIB_CB_FAST_CHECK)
    )
    if not found:
        return None
    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
    )
    m = re.match(r"(\d+)_frame_(\d+)\.png", Path(img_path).name)
    if m is None:
        return None
    return (m.group(1), int(m.group(2))), corners


def detect_all(frames_root: str, camera_ids: list, chessboard: tuple) -> dict:
    """
    Returns dict: cam_id -> {(session, frame_idx): corners}
    """
    n_cores    = max(1, int(mp.cpu_count() * 0.50))
    detections = {}

    for cam_id in camera_ids:
        cam_dir = Path(frames_root) / cam_id
        images  = sorted(cam_dir.glob("*.png"))
        if not images:
            print(f"  [WARN] No frames for cam {cam_id}")
            detections[cam_id] = {}
            continue

        total   = len(images)
        args    = [(str(p), chessboard) for p in images]
        cam_det = {}
        done    = 0

        with mp.Pool(processes=n_cores) as pool:
            for result in pool.imap(_detect_corners_keyed, args):
                done += 1
                if result is not None:
                    key, corners = result
                    cam_det[key] = corners
                fill = int(done / total * 25)
                sys.stdout.write(
                    f"\r  cam {cam_id}: [{'#'*fill}{'-'*(25-fill)}] "
                    f"{done}/{total}  found={len(cam_det)}")
                sys.stdout.flush()
        print()
        detections[cam_id] = cam_det

    return detections


# ─── STEREO CALIBRATION ───────────────────────────────────────────────────────

def stereo_pair(cam_a: str, cam_b: str, det_a: dict, det_b: dict,
                intrinsics: dict, chessboard: tuple, sq_m: float):
    """
    Stereo-calibrate cameras A and B.
    Returns (R, T, rms, n_pairs) where p_camB = R @ p_camA + T, or None.
    """
    shared = sorted(set(det_a.keys()) & set(det_b.keys()))
    if len(shared) < MIN_STEREO_PAIRS:
        print(f"  [{cam_a}↔{cam_b}] only {len(shared)} shared frames — skipping")
        return None

    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= sq_m

    Ka, da = intrinsics[cam_a]["K"], intrinsics[cam_a]["dist"]
    Kb, db = intrinsics[cam_b]["K"], intrinsics[cam_b]["dist"]
    img_sz = tuple(intrinsics[cam_a]["image_size"])

    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        [objp] * len(shared),
        [det_a[k] for k in shared],
        [det_b[k] for k in shared],
        Ka.copy(), da.copy(), Kb.copy(), db.copy(), img_sz,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return R, T.reshape(3, 1), rms, len(shared)


def build_poses(camera_ids: list, reference_cam: str,
                detections: dict, intrinsics: dict,
                chessboard: tuple, sq_m: float) -> tuple:
    """
    Build absolute world-to-camera poses for all cameras.
    Reference camera gets R=I, T=0.  Others computed via stereo calibration,
    with fallback chaining through any already-solved camera.
    Returns (abs_poses dict, cam_n_shared dict).
    """
    abs_poses    = {reference_cam: (np.eye(3), np.zeros((3, 1)))}
    cam_n_shared = {reference_cam: 0}
    remaining    = [c for c in camera_ids if c != reference_cam]

    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        for cam in list(remaining):
            for anchor, (R_anchor, T_anchor) in abs_poses.items():
                result = stereo_pair(anchor, cam,
                                     detections[anchor], detections[cam],
                                     intrinsics, chessboard, sq_m)
                if result is None:
                    continue
                R_rel, T_rel, rms, n_pairs = result
                # p_cam = R_rel @ p_anchor + T_rel
                #       = R_rel @ (R_anchor @ p_world + T_anchor) + T_rel
                R_abs = R_rel @ R_anchor
                T_abs = R_rel @ T_anchor + T_rel
                abs_poses[cam]    = (R_abs, T_abs)
                cam_n_shared[cam] = n_pairs
                remaining.remove(cam)
                print(f"  cam {anchor} -> cam {cam}:  "
                      f"{n_pairs} shared pairs  RMS={rms:.4f} px")
                break

    if remaining:
        print(f"  [WARN] Could not compute poses for: {remaining}"
              f" — insufficient shared detections")

    return abs_poses, cam_n_shared


# ─── REFERENCE POINT TRIANGULATION ───────────────────────────────────────────

def _dlt(P_list: list, pts2d: list) -> np.ndarray:
    rows = []
    for P, (x, y) in zip(P_list, pts2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def build_reference_points(abs_poses: dict, detections: dict,
                            intrinsics: dict, chessboard: tuple,
                            sq_m: float, output_dir: str, n_pts: int = 20):
    """
    Triangulate checkerboard corners visible in all calibrated cameras.
    Saves stereo_points3d.txt — candidates for world_registration.csv.
    """
    cams = list(abs_poses.keys())
    if len(cams) < 2:
        return

    P_map = {}
    for cam_id, (R, T) in abs_poses.items():
        if cam_id in intrinsics:
            K = intrinsics[cam_id]["K"]
            P_map[cam_id] = K @ np.hstack([R, T.reshape(3, 1)])

    common_keys = set(detections[cams[0]].keys())
    for c in cams[1:]:
        common_keys &= set(detections[c].keys())
    common_keys = sorted(common_keys)

    if not common_keys:
        print("  [INFO] No frames with all cameras detecting the board — "
              "reference points not generated")
        return

    cam_list = [c for c in cams if c in P_map]
    P_list   = [P_map[c] for c in cam_list]

    frame_step    = max(1, len(common_keys) // 5)
    corner_stride = max(1, (chessboard[0] * chessboard[1]) // 5)
    corner_idxs   = list(range(0, chessboard[0] * chessboard[1], corner_stride))

    pts3d = []
    for frame_key in common_keys[::frame_step]:
        for ci in corner_idxs:
            obs = []
            ok  = True
            for cam_id in cam_list:
                corners = detections[cam_id].get(frame_key)
                if corners is None or ci >= len(corners):
                    ok = False
                    break
                obs.append(corners[ci, 0, :])
            if not ok:
                continue
            p3d = _dlt(P_list, obs)
            pts3d.append((f"{frame_key[0]}_f{frame_key[1]:06d}_c{ci:03d}", p3d))
            if len(pts3d) >= n_pts:
                break
        if len(pts3d) >= n_pts:
            break

    if not pts3d:
        return

    out_path = Path(output_dir) / "stereo_points3d.txt"
    with open(out_path, "w") as f:
        f.write(f"# Triangulated checkerboard corners in cam {REFERENCE_CAM} coords\n")
        f.write("# name  x  y  z\n")
        for name, p in pts3d:
            f.write(f"{name}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n")

    print(f"\n  Reference candidates saved -> {out_path}")
    print(f"  {'NAME':36s}  {'X':>10}  {'Y':>10}  {'Z':>10}")
    for name, p in pts3d:
        print(f"  {name:36s}  {p[0]:>10.4f}  {p[1]:>10.4f}  {p[2]:>10.4f}")
    print(f"\n  Use these X/Y/Z values in world_registration.csv (colmap_x/y/z columns).")
    print(f"  Pick 3+ points whose real-world positions you can measure in mm.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _setup_log(str(Path(__file__).parent / "run.log"))
    print("=" * 60)
    print("DATALUS — STEP 3: MULTI-CAMERA STEREO CALIBRATION")
    print("=" * 60)

    if not os.path.exists(INTRINSICS_NPZ):
        print(f"[ERROR] {INTRINSICS_NPZ} not found. Run Step 2 first.")
        sys.exit(1)

    # Load intrinsics
    data       = np.load(INTRINSICS_NPZ)
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        if f"{cam_id}_K" in data:
            intrinsics[cam_id] = {
                "K":          data[f"{cam_id}_K"],
                "dist":       data[f"{cam_id}_dist"],
                "image_size": tuple(int(v) for v in data[f"{cam_id}_image_size"]),
            }
        else:
            print(f"  [WARN] No intrinsics for cam {cam_id}")

    sq_m = SQUARE_SIZE_MM / 1000.0

    print(f"\n  Reference camera: {REFERENCE_CAM}")

    det_path = Path(OUTPUT_DIR) / "detections.npz"
    if det_path.exists():
        print(f"\n  Loading detections from Step 2 -> {det_path}")
        raw        = np.load(str(det_path))
        detections = {}
        for cam_id in CAMERA_IDS:
            if f"{cam_id}_corners" not in raw:
                detections[cam_id] = {}
                continue
            sessions = raw[f"{cam_id}_sessions"]
            frames   = raw[f"{cam_id}_frames"]
            corners  = raw[f"{cam_id}_corners"]   # (N, N_corners, 2)
            detections[cam_id] = {
                (str(sessions[i]), int(frames[i])): corners[i].reshape(-1, 1, 2)
                for i in range(len(sessions))
            }
            print(f"    cam {cam_id}: {len(detections[cam_id])} detections loaded")
    else:
        print(f"\n  detections.npz not found — detecting corners from scratch...")
        detections = detect_all(FRAMES_ROOT, CAMERA_IDS, CHESSBOARD)

    print(f"\n  Building pose graph...")
    abs_poses, cam_n_shared = build_poses(
        CAMERA_IDS, REFERENCE_CAM, detections, intrinsics, CHESSBOARD, sq_m
    )

    if len(abs_poses) < 2:
        print("[ERROR] Could not compute poses for enough cameras.")
        sys.exit(1)

    build_reference_points(abs_poses, detections, intrinsics,
                           CHESSBOARD, sq_m, OUTPUT_DIR)

    # Save poses (same format as old colmap_poses.npz — compatible with Step 4)
    out_path  = Path(OUTPUT_DIR) / "colmap_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in abs_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
        save_dict[f"{cam_id}_n"] = np.array([cam_n_shared.get(cam_id, 0)])
    np.savez(str(out_path), **save_dict)
    print(f"\n  Poses saved -> {out_path}")

    print("\nSummary:")
    for cam_id, (R, T) in abs_poses.items():
        C = -R.T @ T
        n = cam_n_shared.get(cam_id, 0)
        label = f"{n} shared pairs" if n > 0 else "reference"
        print(f"  cam {cam_id}: centre=[{C[0,0]:.4f}, {C[1,0]:.4f}, {C[2,0]:.4f}]"
              f"  ({label})")

    print("\nStep 3 complete.")
    print("Next: fill in world_registration.csv using stereo_points3d.txt,")
    print("      then run Step 4 (world registration).")


if __name__ == "__main__":
    main()
