#!/usr/bin/env python3
"""
Datalus Calibration — Full Pipeline (Steps 1-6)

Steps:
  1. Frame extraction        — sample one frame per 500 ms from all 8 videos
  2. Intrinsic calibration   — cv2.calibrateCamera with spec-derived focal prior
  3. Stereo calibration      — relative poses via cv2.stereoCalibrate
  4. World registration      — Umeyama similarity transform (COLMAP → real mm)
  5. YAML export             — one ABT-compatible YAML per camera
  6. Validation              — intrinsic RMS + world residuals + DLT cross-check

Usage:
    python DatulusCalib_Full.py              # run full pipeline
    python DatulusCalib_Full.py --from 3     # resume from step N (1-6)

Step 4 will pause if world_registration.csv is not ready, then exit with
instructions. Re-run with --from 4 once the CSV is filled in.
"""

import argparse
import csv
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import multiprocessing as mp
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────────

SESSIONS = {
    "250707": {
        "102": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4",
        "108": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4",
        "113": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4",
        "117": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4",
    },
    "250708": {
        "102": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250708/_2_102_20250708161657.mp4",
        "108": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250708/_2_108_20250708161657.mp4",
        "113": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250708/_2_113_20250708161657.mp4",
        "117": "/Users/acalapai/PycharmProjects/Datalus/CalibrationVideos/250708/_2_117_20250708161657.mp4",
    },
}

CAMERA_IDS         = ["102", "108", "113", "117"]

CHESSBOARD         = (13, 9)      # inner corners (width × height)
SQUARE_SIZE_MM     = 40.0         # physical square size in mm

SENSOR_WIDTH_MM    = 11.2         # Sony IMX304 1.1" sensor
FOCAL_LENGTH_MM    = 8.0          # C-mount lens focal length

OUTPUT_DIR         = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration"
WORLD_CSV          = os.path.join(OUTPUT_DIR, "world_registration.csv")

FRAME_INTERVAL_MS  = 500     # sample one frame per 500 ms
REFERENCE_CAM      = "102"   # all extrinsics expressed relative to this camera
MIN_STEREO_PAIRS   = 10      # minimum shared detections required per camera pair

PNG_PARAMS         = [cv2.IMWRITE_PNG_COMPRESSION, 0]   # fastest write
UPDATE_EVERY       = 200          # frame-extraction progress interval

# ─── SHARED UTILITIES ──────────────────────────────────────────────────────────

def _header(title: str):
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─── STEP 1: FRAME EXTRACTION ──────────────────────────────────────────────────

_worker_queue = None  # set per-process by Pool initializer


def _init_worker(q):
    global _worker_queue
    _worker_queue = q


def _extract_video(args):
    """
    Worker: extract frames at the given interval from one video.
    Sends (label, current_frame, total_frames, saved_count) progress to queue.
    Returns (session, cam_id, width, height, saved_count).
    """
    session, cam_id, video_path, cam_dir_str, interval_ms = args
    label   = f"{session}/cam{cam_id}"
    cam_dir = Path(cam_dir_str)
    cam_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _worker_queue.put((label, 0, 1, 0))
        return session, cam_id, None, None, -1

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = 1 if interval_ms == 0 else max(1, int(round(fps * interval_ms / 1000.0)))

    _worker_queue.put((label, 0, total, 0))

    idx = saved = 0
    while True:
        if idx % step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            fname = cam_dir / f"{session}_frame_{idx:06d}.png"
            cv2.imwrite(str(fname), frame, PNG_PARAMS)
            saved += 1
        else:
            if not cap.grab():
                break
        idx += 1
        if idx % UPDATE_EVERY == 0:
            _worker_queue.put((label, idx, total, saved))

    cap.release()
    _worker_queue.put((label, total, total, saved))
    return session, cam_id, w, h, saved


def _render_progress(state: dict, labels: list) -> str:
    done  = sum(1 for l in labels if state[l][0] >= state[l][1] > 1)
    parts = []
    for lbl in labels:
        cur, tot, _ = state[lbl]
        if cur >= tot > 1:
            parts.append(f"{lbl}:done")
        else:
            parts.append(f"{lbl}:{int(min(1.0, cur / max(tot,1)) * 100):3d}%")
    return f"  [{done}/{len(labels)}] " + "  ".join(parts)


def _monitor(queue, labels: list, stop_evt: threading.Event):
    """Background thread: rewrites a single status line every 150 ms."""
    state = {l: (0, 1, 0) for l in labels}
    while not stop_evt.is_set():
        try:
            while True:
                lbl, cur, tot, sav = queue.get_nowait()
                state[lbl] = (cur, tot, sav)
        except Exception:
            pass
        sys.stdout.write(f"\r{_render_progress(state, labels)}  ")
        sys.stdout.flush()
        time.sleep(0.15)
    # final drain
    try:
        while True:
            lbl, cur, tot, sav = queue.get_nowait()
            state[lbl] = (cur, tot, sav)
    except Exception:
        pass
    sys.stdout.write(f"\r{_render_progress(state, labels)}  \n")
    sys.stdout.flush()


def step1_extract_frames(sessions: dict, output_dir: str, interval_ms: int):
    """
    Extract frames from all videos in parallel.
    Returns (frames_root Path, image_size dict: cam_id -> (w, h)).
    """
    _header("STEP 1 — FRAME EXTRACTION")
    frames_root = Path(output_dir) / "frames"

    jobs = []
    for session, cams in sessions.items():
        for cam_id, vpath in cams.items():
            if not os.path.exists(vpath):
                print(f"[ERROR] Video not found: {vpath}")
                sys.exit(1)
            jobs.append((session, cam_id, vpath,
                         str(frames_root / cam_id), interval_ms))

    labels    = [f"{s}/cam{c}" for s, c, *_ in jobs]
    n_workers = min(len(jobs), mp.cpu_count())
    print(f"\n  {len(jobs)} videos — {n_workers} parallel workers"
          f"  (interval={interval_ms} ms)")

    queue    = mp.Queue()
    stop_evt = threading.Event()
    mon      = threading.Thread(target=_monitor, args=(queue, labels, stop_evt),
                                daemon=True)
    mon.start()

    image_size = {}
    with mp.Pool(processes=n_workers, initializer=_init_worker,
                 initargs=(queue,)) as pool:
        for session, cam_id, w, h, saved in pool.imap_unordered(_extract_video, jobs):
            if saved == -1:
                stop_evt.set()
                mon.join()
                print(f"\n[ERROR] Cannot open video for [{session}] cam {cam_id}")
                sys.exit(1)
            if cam_id not in image_size and w is not None:
                image_size[cam_id] = (w, h)

    stop_evt.set()
    mon.join()

    print("\n  Resolutions detected:")
    for cam_id, (w, h) in sorted(image_size.items()):
        f_px = FOCAL_LENGTH_MM * w / SENSOR_WIDTH_MM
        print(f"    cam {cam_id}: {w}x{h}  ->  prior focal = {f_px:.1f} px")

    print("\n  Step 1 complete.")
    return frames_root, image_size


# ─── STEP 2: INTRINSIC CALIBRATION ─────────────────────────────────────────────

def _detect_corners(args):
    """Worker: detect and sub-pixel refine checkerboard corners in one frame."""
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
    if found:
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)
        )
        return corners
    return None


def step2_calibrate_intrinsics(frames_root, camera_ids: list,
                                chessboard: tuple, square_size_mm: float,
                                focal_mm: float, sensor_w_mm: float,
                                output_dir: str) -> dict:
    """
    Detect checkerboard corners and run cv2.calibrateCamera per camera.
    Returns intrinsics dict: cam_id -> {K, dist, rms, n_frames, image_size}.
    """
    _header("STEP 2 — INTRINSIC CALIBRATION")

    sq_m = square_size_mm / 1000.0
    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= sq_m

    n_cores    = max(1, int(mp.cpu_count() * 0.75))
    intrinsics = {}
    all_det    = {}   # cam_id -> (sessions[], frames[], corners[])

    for cam_id in camera_ids:
        cam_dir = Path(frames_root) / cam_id
        images  = sorted(cam_dir.glob("*.png"))
        if not images:
            print(f"\n  [WARN] No frames for cam {cam_id} in {cam_dir}")
            continue

        sample = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
        if sample is None:
            print(f"\n  [ERROR] Cannot read {images[0]}")
            sys.exit(1)
        h, w  = sample.shape
        f_px  = focal_mm * w / sensor_w_mm
        total = len(images)

        print(f"\n  cam {cam_id}: {w}x{h}  prior focal={f_px:.1f}px  "
              f"frames={total}  cores={n_cores}")

        obj_pts, img_pts = [], []
        det_sessions, det_frames, det_corners = [], [], []
        completed = 0
        args_list = [(str(p), chessboard) for p in images]

        with mp.Pool(processes=n_cores) as pool:
            for idx, result in enumerate(pool.imap(_detect_corners, args_list)):
                completed += 1
                if result is not None:
                    obj_pts.append(objp)
                    img_pts.append(result)
                    m = re.match(r"(\d+)_frame_(\d+)\.png", images[idx].name)
                    if m:
                        det_sessions.append(int(m.group(1)))
                        det_frames.append(int(m.group(2)))
                        det_corners.append(result[:, 0, :])  # (N_corners, 2)
                fill = int(completed / total * 30)
                sys.stdout.write(
                    f"\r    [{'#'*fill}{'-'*(30-fill)}] "
                    f"{completed}/{total}  found={len(obj_pts)}")
                sys.stdout.flush()
        print()
        all_det[cam_id] = (det_sessions, det_frames, det_corners)

        if len(obj_pts) < 3:
            print(f"    [WARN] Too few detections ({len(obj_pts)}) for cam {cam_id} — skipping")
            continue

        K_init = np.array(
            [[f_px, 0,    w / 2.0],
             [0,    f_px, h / 2.0],
             [0,    0,    1.0   ]], dtype=np.float64)

        rms, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, (w, h), K_init.copy(), None,
            flags=(cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3)
        )

        d = np.zeros(5)
        d[:min(5, dist.ravel().size)] = dist.ravel()[:5]

        intrinsics[cam_id] = {
            "K":          K,
            "dist":       d.reshape(1, 5),
            "rms":        rms,
            "n_frames":   len(obj_pts),
            "image_size": (w, h),
        }
        print(f"    RMS={rms:.4f}px  "
              f"fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  "
              f"cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
        print(f"    dist={d}")

    if not intrinsics:
        print("[ERROR] No cameras calibrated. Check FRAMES_ROOT and checkerboard config.")
        sys.exit(1)

    npz_path  = Path(output_dir) / "intrinsics.npz"
    save_dict = {}
    for cam_id, data in intrinsics.items():
        save_dict[f"{cam_id}_K"]          = data["K"]
        save_dict[f"{cam_id}_dist"]       = data["dist"]
        save_dict[f"{cam_id}_rms"]        = np.array([data["rms"]])
        save_dict[f"{cam_id}_image_size"] = np.array(data["image_size"])
    np.savez(str(npz_path), **save_dict)
    print(f"\n  Intrinsics saved -> {npz_path}")

    # Save detections so Step 3 can load them without re-detecting
    det_path  = Path(output_dir) / "detections.npz"
    det_dict  = {}
    for cam_id, (sessions, frames, corners) in all_det.items():
        if corners:
            det_dict[f"{cam_id}_sessions"] = np.array(sessions, dtype=np.int32)
            det_dict[f"{cam_id}_frames"]   = np.array(frames,   dtype=np.int32)
            det_dict[f"{cam_id}_corners"]  = np.array(corners,  dtype=np.float32)
    np.savez(str(det_path), **det_dict)
    print(f"  Detections saved -> {det_path}")

    print("\n  Step 2 complete.")
    return intrinsics


def _load_intrinsics(output_dir: str, camera_ids: list) -> dict:
    npz_path = Path(output_dir) / "intrinsics.npz"
    if not npz_path.exists():
        print(f"[ERROR] {npz_path} not found. Run step 2 first.")
        sys.exit(1)
    data       = np.load(str(npz_path))
    intrinsics = {}
    for cam_id in camera_ids:
        if f"{cam_id}_K" in data:
            intrinsics[cam_id] = {
                "K":          data[f"{cam_id}_K"],
                "dist":       data[f"{cam_id}_dist"],
                "rms":        float(data[f"{cam_id}_rms"][0])
                              if f"{cam_id}_rms" in data else 0.0,
                "n_frames":   0,
                "image_size": tuple(int(v) for v in data[f"{cam_id}_image_size"])
                              if f"{cam_id}_image_size" in data else (0, 0),
            }
    return intrinsics


# ─── STEP 3: MULTI-CAMERA STEREO CALIBRATION ───────────────────────────────────

def _detect_corners_keyed(args):
    """
    Worker: detect checkerboard corners and return ((session, frame_idx), corners).
    Returns None if not found or filename doesn't match expected pattern.
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


def _detect_all(frames_root: Path, camera_ids: list, chessboard: tuple,
                n_cores: int) -> dict:
    """
    Detect corners in all frames for all cameras.
    Returns dict: cam_id -> {(session, frame_idx): corners}
    """
    detections = {}
    for cam_id in camera_ids:
        cam_dir = frames_root / cam_id
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


def _stereo_pair(cam_a: str, cam_b: str, det_a: dict, det_b: dict,
                 intrinsics: dict, chessboard: tuple, sq_m: float):
    """
    Stereo-calibrate cameras A and B using shared checkerboard detections.
    Returns (R, T, rms, n_pairs) where p_camB = R @ p_camA + T, or None.
    """
    shared = sorted(set(det_a.keys()) & set(det_b.keys()))
    if len(shared) < MIN_STEREO_PAIRS:
        return None

    objp = np.zeros((chessboard[0] * chessboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    objp *= sq_m

    obj_pts   = [objp] * len(shared)
    img_pts_a = [det_a[k] for k in shared]
    img_pts_b = [det_b[k] for k in shared]

    Ka, da = intrinsics[cam_a]["K"], intrinsics[cam_a]["dist"]
    Kb, db = intrinsics[cam_b]["K"], intrinsics[cam_b]["dist"]
    img_sz = tuple(intrinsics[cam_a]["image_size"])

    rms, *_, R, T, _, _ = cv2.stereoCalibrate(
        obj_pts, img_pts_a, img_pts_b,
        Ka.copy(), da.copy(), Kb.copy(), db.copy(), img_sz,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return R, T.reshape(3, 1), rms, len(shared)


def _triangulate_dlt_s3(P_list: list, pts2d: list) -> np.ndarray:
    """DLT triangulation from ≥2 views. Returns (3,) world point."""
    rows = []
    for P, (x, y) in zip(P_list, pts2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def _build_reference_points(abs_poses: dict, detections: dict,
                             intrinsics: dict, chessboard: tuple,
                             sq_m: float, output_dir: str, n_pts: int = 20):
    """
    Triangulate checkerboard corners visible in all calibrated cameras.
    Saves stereo_points3d.txt — candidates for filling world_registration.csv.
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

    # Sample a spread of corners across several frames
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
            p3d = _triangulate_dlt_s3(P_list, obs)
            pts3d.append((f"{frame_key[0]}_f{frame_key[1]:06d}_c{ci:03d}", p3d))
            if len(pts3d) >= n_pts:
                break
        if len(pts3d) >= n_pts:
            break

    if not pts3d:
        return

    out_path = Path(output_dir) / "stereo_points3d.txt"
    with open(out_path, "w") as f:
        f.write("# Triangulated checkerboard corners in reference-camera (cam "
                f"{REFERENCE_CAM}) coordinates\n")
        f.write("# name  x  y  z\n")
        for name, p in pts3d:
            f.write(f"{name}  {p[0]:.6f}  {p[1]:.6f}  {p[2]:.6f}\n")

    print(f"\n  Triangulated reference candidates -> {out_path}")
    print(f"  {'NAME':36s}  {'X':>10}  {'Y':>10}  {'Z':>10}")
    for name, p in pts3d:
        print(f"  {name:36s}  {p[0]:>10.4f}  {p[1]:>10.4f}  {p[2]:>10.4f}")


def step3_stereo(frames_root: Path, intrinsics: dict, camera_ids: list,
                 output_dir: str):
    """
    Compute per-camera extrinsics via pairwise cv2.stereoCalibrate on
    synchronized checkerboard detections.  All poses are expressed relative
    to REFERENCE_CAM (R=I, T=0).
    Returns (cam_poses dict, cam_n_shared dict).
    """
    _header("STEP 3 — MULTI-CAMERA STEREO CALIBRATION")
    sq_m    = SQUARE_SIZE_MM / 1000.0
    n_cores = max(1, int(mp.cpu_count() * 0.75))

    print(f"\n  Reference camera: {REFERENCE_CAM}")

    det_path = Path(output_dir) / "detections.npz"
    if det_path.exists():
        print(f"\n  Loading detections from Step 2 -> {det_path}")
        raw = np.load(str(det_path))
        detections = {}
        for cam_id in camera_ids:
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
        print(f"\n  detections.npz not found — re-detecting corners...")
        detections = _detect_all(frames_root, camera_ids, CHESSBOARD, n_cores)

    print(f"\n  Building pose graph...")
    abs_poses    = {REFERENCE_CAM: (np.eye(3), np.zeros((3, 1)))}
    cam_n_shared = {REFERENCE_CAM: 0}
    remaining    = [c for c in camera_ids if c != REFERENCE_CAM]

    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        for cam in list(remaining):
            for anchor, (R_anchor, T_anchor) in abs_poses.items():
                result = _stereo_pair(anchor, cam,
                                      detections[anchor], detections[cam],
                                      intrinsics, CHESSBOARD, sq_m)
                if result is None:
                    continue
                R_rel, T_rel, rms, n_pairs = result
                # chain: p_cam = R_rel @ p_anchor + T_rel
                #              = R_rel @ (R_anchor @ p_world + T_anchor) + T_rel
                R_abs = R_rel @ R_anchor
                T_abs = R_rel @ T_anchor + T_rel
                abs_poses[cam]    = (R_abs, T_abs)
                cam_n_shared[cam] = n_pairs
                remaining.remove(cam)
                print(f"    cam {anchor} -> cam {cam}:  "
                      f"{n_pairs} shared pairs  RMS={rms:.4f} px")
                break

    if remaining:
        print(f"  [WARN] Could not compute poses for: {remaining} "
              f"(insufficient shared detections)")

    _build_reference_points(abs_poses, detections, intrinsics,
                            CHESSBOARD, sq_m, output_dir)

    npz_path  = Path(output_dir) / "colmap_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in abs_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
        save_dict[f"{cam_id}_n"] = np.array([cam_n_shared.get(cam_id, 0)])
    np.savez(str(npz_path), **save_dict)
    print(f"\n  Poses saved -> {npz_path}")

    print("\nSummary:")
    for cam_id, (R, T) in abs_poses.items():
        C = -R.T @ T
        print(f"  cam {cam_id}: centre = "
              f"[{C[0,0]:.4f}, {C[1,0]:.4f}, {C[2,0]:.4f}]"
              f"  (in cam {REFERENCE_CAM} frame)")

    print("\n  Step 3 complete.")
    return abs_poses, cam_n_shared


def _load_colmap_poses(output_dir: str, camera_ids: list):
    npz_path = Path(output_dir) / "colmap_poses.npz"
    if not npz_path.exists():
        print(f"[ERROR] {npz_path} not found. Run step 3 first.")
        sys.exit(1)
    data         = np.load(str(npz_path))
    poses        = {}
    cam_n_frames = {}
    for cam_id in camera_ids:
        if f"{cam_id}_R" in data:
            poses[cam_id]        = (data[f"{cam_id}_R"], data[f"{cam_id}_T"])
            cam_n_frames[cam_id] = int(data[f"{cam_id}_n"][0]) \
                                   if f"{cam_id}_n" in data else 0
    return poses, cam_n_frames


# ─── STEP 4: WORLD REGISTRATION ────────────────────────────────────────────────

def _umeyama(src: np.ndarray, dst: np.ndarray):
    """
    Umeyama (1991) similarity transform: dst ≈ s * R @ src + t
    src, dst : (N, 3) float64, N >= 3
    Returns  : s (float), R (3x3), t (3,)
    """
    n    = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    src_c, dst_c = src - mu_s, dst - mu_d
    sigma_s2 = (src_c ** 2).sum() / n
    Sigma    = (dst_c.T @ src_c) / n
    U, d, Vt = np.linalg.svd(Sigma)
    S        = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = np.trace(np.diag(d) @ S) / sigma_s2
    t = mu_d - s * R @ mu_s
    return s, R, t


def _transform_pose(R_c: np.ndarray, T_c: np.ndarray,
                    s: float, R_w: np.ndarray, t_w: np.ndarray):
    """
    Convert COLMAP world-to-camera pose (R_c, T_c) to world-mm pose.

    COLMAP:   p_cam = R_c @ p_colmap + T_c
    World:    p_world_mm = s * R_w @ p_colmap + t_w

    Result:   p_cam = R_out @ p_world_mm + T_out
              R_out = R_c @ R_w.T
              T_out = s * T_c - R_out @ t_w
    """
    R_out = R_c @ R_w.T
    T_out = s * T_c - R_out @ t_w.reshape(3, 1)
    return R_out, T_out


def _load_world_csv(csv_path: str):
    """
    Load world_registration.csv.
    Returns src (N,3) COLMAP XYZ, dst (N,3) real-world mm, names list.
    """
    src, dst, names = [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            x = row["colmap_x"].strip()
            if not x or x.startswith("#"):
                continue
            names.append(row["name"].strip())
            src.append([float(row["colmap_x"]),
                        float(row["colmap_y"]),
                        float(row["colmap_z"])])
            dst.append([float(row["real_x_mm"]),
                        float(row["real_y_mm"]),
                        float(row["real_z_mm"])])
    return np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64), names


def _create_csv_template(csv_path: str, points3d_txt: str):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "colmap_x", "colmap_y", "colmap_z",
                    "real_x_mm", "real_y_mm", "real_z_mm"])
        for name in ["point_A", "point_B", "point_C"]:
            w.writerow([name, "", "", "", "", "", ""])
    print(f"\n  Template created: {csv_path}")

    if not os.path.exists(points3d_txt):
        print(f"  (stereo_points3d.txt not found — run Step 3 first)")
        return

    print(f"\n  Triangulated reference candidates (copy X/Y/Z into colmap_x/y/z):")
    print(f"  {'NAME':36s}  {'X':>10}  {'Y':>10}  {'Z':>10}")
    with open(points3d_txt) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            if len(p) >= 4:
                print(f"  {p[0]:36s}  {float(p[1]):>10.4f}  "
                      f"{float(p[2]):>10.4f}  {float(p[3]):>10.4f}")


def step4_world_registration(colmap_poses: dict, output_dir: str,
                              world_csv: str, camera_ids: list):
    """
    Fit Umeyama similarity transform from COLMAP to real-world mm.
    Returns (world_poses, s, R_w, t_w) or (None, None, None, None) if CSV not ready.
    """
    _header("STEP 4 — WORLD REGISTRATION")

    points3d_txt = str(Path(output_dir) / "stereo_points3d.txt")

    # Check CSV readiness
    csv_ready = False
    if os.path.exists(world_csv):
        try:
            src, dst, names = _load_world_csv(world_csv)
            csv_ready = len(names) >= 3
        except Exception:
            pass

    if not csv_ready:
        print("\n  world_registration.csv not ready (missing, empty, or < 3 points).")
        _create_csv_template(world_csv, points3d_txt)
        print("\n  Fill in the CSV, then re-run with --from 4")
        return None, None, None, None

    src, dst, names = _load_world_csv(world_csv)
    print(f"\n  Loaded {len(names)} reference points: {names}")

    s, R_w, t_w = _umeyama(src, dst)
    print(f"\n  Umeyama fit:")
    print(f"    scale = {s:.6f}  ({s:.4f} mm per COLMAP unit)")
    for row in R_w:
        print(f"    R_w   [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
    print(f"    t_w   [{t_w[0]:.3f}, {t_w[1]:.3f}, {t_w[2]:.3f}] mm")

    residuals = []
    print(f"\n  Registration residuals (mm):")
    for i, name in enumerate(names):
        p_est = s * R_w @ src[i] + t_w
        err   = np.linalg.norm(p_est - dst[i])
        residuals.append(err)
        print(f"    {name:20s}  "
              f"est=[{p_est[0]:.2f},{p_est[1]:.2f},{p_est[2]:.2f}]  "
              f"gt=[{dst[i,0]:.2f},{dst[i,1]:.2f},{dst[i,2]:.2f}]  "
              f"err={err:.3f}mm")
    rmse = np.sqrt(np.mean(np.array(residuals) ** 2))
    print(f"    RMSE = {rmse:.3f} mm")

    world_poses = {}
    print(f"\n  Camera centres in world (mm):")
    for cam_id, (R_c, T_c) in colmap_poses.items():
        R_out, T_out        = _transform_pose(R_c, T_c, s, R_w, t_w)
        world_poses[cam_id] = (R_out, T_out)
        C = -R_out.T @ T_out
        print(f"    cam {cam_id}: [{C[0,0]:.2f}, {C[1,0]:.2f}, {C[2,0]:.2f}] mm")

    npz_path  = Path(output_dir) / "world_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in world_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
    np.savez(str(npz_path), **save_dict)
    print(f"\n  World poses saved -> {npz_path}")
    print("\n  Step 4 complete.")
    return world_poses, s, R_w, t_w


def _load_world_poses(output_dir: str, camera_ids: list) -> dict:
    npz_path = Path(output_dir) / "world_poses.npz"
    if not npz_path.exists():
        print(f"[ERROR] {npz_path} not found. Run step 4 first.")
        sys.exit(1)
    data  = np.load(str(npz_path))
    poses = {}
    for cam_id in camera_ids:
        if f"{cam_id}_R" in data:
            poses[cam_id] = (data[f"{cam_id}_R"], data[f"{cam_id}_T"])
    return poses


# ─── STEP 5: YAML EXPORT ───────────────────────────────────────────────────────
#
# ABT's utils.read_camera_parameters reads each matrix as a flat array, reshapes
# to 3×3 row-major, then TRANSPOSES.  Therefore we must store K^T and R^T so
# that after the transpose ABT gets back the standard K and R.
# T is read directly as a (3,1) column vector — no transposition — so it is
# stored as-is.
# Distortion coefficients are stored as a (1,5) row vector: [k1,k2,p1,p2,k3].

def _fmt_val(v: float) -> str:
    """Format a float in OpenCV FileStorage style: '0.' for zero, scientific for others."""
    if v == 0.0:
        return "0."
    iv = int(v)
    if float(iv) == v:
        return f"{iv}."
    return f"{v:.16e}"


def _write_ocv_matrix(f, key: str, rows: int, cols: int, data_flat):
    """
    Write one !!opencv-matrix block, matching the exact whitespace of OpenCV
    FileStorage so ABT's preprocess_yaml / yaml.safe_load can parse it.
    """
    vals = [_fmt_val(float(v)) for v in data_flat]
    f.write(f"{key}: !!opencv-matrix\n")
    f.write(f"   rows: {rows}\n")
    f.write(f"   cols: {cols}\n")
    f.write(f"   dt: d\n")
    if len(vals) <= 5:
        f.write(f"   data: [ {', '.join(vals)} ]\n")
    else:
        # first line holds 5 values, continuation lines hold up to 5
        f.write(f"   data: [ {', '.join(vals[:5])},\n")
        remaining = vals[5:]
        while remaining:
            chunk     = remaining[:5]
            remaining = remaining[5:]
            sep       = "," if remaining else " ]"
            f.write(f"       {', '.join(chunk)}{sep}\n")


def step5_export_yaml(world_poses: dict, intrinsics: dict,
                      output_dir: str, camera_ids: list):
    """
    Write one ABT-compatible YAML file per camera to output_dir/cam_{id}.yaml.
    """
    _header("STEP 5 — YAML EXPORT")

    out_dir = Path(output_dir)
    written = []

    for cam_id in camera_ids:
        if cam_id not in world_poses:
            print(f"  [SKIP] cam {cam_id}: no world pose")
            continue
        if cam_id not in intrinsics:
            print(f"  [SKIP] cam {cam_id}: no intrinsics")
            continue

        K    = intrinsics[cam_id]["K"]            # (3,3)
        dist = intrinsics[cam_id]["dist"].ravel() # (5,)
        R, T = world_poses[cam_id]                # (3,3), (3,1)

        # ABT transposes both K and R when loading → store K^T and R^T row-major
        K_T   = K.T.ravel()          # [fx,0,0, 0,fy,0, cx,cy,1]
        R_T   = R.T.ravel()          # R^T stored row-major
        T_col = T.ravel()            # [tx, ty, tz] — no transpose

        yaml_path = out_dir / f"cam_{cam_id}.yaml"
        with open(yaml_path, "w") as f:
            f.write("%YAML:1.0\n---\n")
            _write_ocv_matrix(f, "intrinsicMatrix",        3, 3, K_T)
            _write_ocv_matrix(f, "distortionCoefficients", 1, 5, dist)
            _write_ocv_matrix(f, "R",                      3, 3, R_T)
            _write_ocv_matrix(f, "T",                      3, 1, T_col)

        written.append(yaml_path.name)
        print(f"  cam {cam_id}  ->  {yaml_path}")

    print(f"\n  {len(written)} YAML files written: {written}")
    print("\n  Step 5 complete.")


# ─── STEP 6: VALIDATION ────────────────────────────────────────────────────────

def _dlt_triangulate(P_list: list, pts2d: list) -> np.ndarray:
    """
    Linear (DLT) triangulation from N >= 2 cameras.
    P_list : list of (3,4) projection matrices
    pts2d  : list of (2,) pixel observations (one per camera)
    Returns (3,) world point.
    """
    rows = []
    for P, (x, y) in zip(P_list, pts2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A         = np.array(rows)
    _, _, Vt  = np.linalg.svd(A)
    X         = Vt[-1]
    return X[:3] / X[3]


def step6_validate(intrinsics: dict, world_poses: dict, world_csv: str,
                   cam_n_frames: dict, camera_ids: list):
    """
    Report:
      1. Per-camera intrinsic RMS (from cv2.calibrateCamera)
      2. COLMAP registration frame counts
      3. World registration residuals (Umeyama fit)
      4. DLT round-trip triangulation error using world-calibrated cameras
    """
    _header("STEP 6 — VALIDATION REPORT")

    # ── 1. Intrinsic calibration RMS ──
    print("\n  [1] Intrinsic calibration RMS (per camera):")
    for cam_id in camera_ids:
        if cam_id in intrinsics:
            d = intrinsics[cam_id]
            print(f"      cam {cam_id}:  RMS = {d['rms']:.4f} px"
                  f"  ({d.get('n_frames', '?')} frames used)")

    # ── 2. Stereo calibration coverage ──
    if cam_n_frames:
        print("\n  [2] Stereo calibration shared detections per camera:")
        for cam_id in camera_ids:
            if cam_id in cam_n_frames:
                n = cam_n_frames[cam_id]
                label = f"{n} shared pairs with ref cam" if n > 0 else "reference camera"
                print(f"      cam {cam_id}:  {label}")

    # ── 3. World registration residuals ──
    if not os.path.exists(world_csv):
        print("\n  [3] World CSV not found — skipping world residuals.")
        return
    try:
        src, dst, names = _load_world_csv(world_csv)
    except Exception as e:
        print(f"\n  [3] Could not load CSV: {e}")
        return
    if len(names) < 3:
        print("\n  [3] Fewer than 3 reference points — skipping.")
        return

    s, R_w, t_w = _umeyama(src, dst)
    residuals   = []
    print(f"\n  [3] World registration residuals (Umeyama):")
    for i, name in enumerate(names):
        p_est = s * R_w @ src[i] + t_w
        err   = np.linalg.norm(p_est - dst[i])
        residuals.append(err)
        print(f"      {name:20s}  err = {err:.3f} mm")
    rmse = np.sqrt(np.mean(np.array(residuals) ** 2))
    print(f"      RMSE = {rmse:.3f} mm")

    # ── 4. DLT round-trip triangulation ──
    # Build projection matrices P_i = K_i @ [R_i | T_i] in world-mm space
    P_map = {}
    for cam_id in camera_ids:
        if cam_id not in world_poses or cam_id not in intrinsics:
            continue
        K    = intrinsics[cam_id]["K"]
        R, T = world_poses[cam_id]
        P_map[cam_id] = K @ np.hstack([R, T.reshape(3, 1)])

    if len(P_map) < 2:
        print("\n  [4] Need >= 2 calibrated cameras for triangulation check — skipping.")
        return

    cam_list = list(P_map.keys())
    P_list   = [P_map[c] for c in cam_list]

    print(f"\n  [4] DLT triangulation round-trip (project known 3D -> triangulate -> compare):")
    print(f"      Cameras used: {cam_list}")
    tri_errors = []
    for i, name in enumerate(names):
        P_gt  = dst[i]   # ground-truth world position in mm
        # Project through each camera to get synthetic 2D observations
        pts2d = []
        for P in P_list:
            x_h  = np.append(P_gt, 1.0)
            proj = P @ x_h
            pts2d.append((proj[0] / proj[2], proj[1] / proj[2]))
        # Triangulate back
        P_tri = _dlt_triangulate(P_list, pts2d)
        err   = np.linalg.norm(P_tri - P_gt)
        tri_errors.append(err)
        print(f"      {name:20s}  "
              f"tri=[{P_tri[0]:.3f},{P_tri[1]:.3f},{P_tri[2]:.3f}]  "
              f"gt=[{P_gt[0]:.3f},{P_gt[1]:.3f},{P_gt[2]:.3f}]  "
              f"err={err:.4f} mm")
    tri_rmse = np.sqrt(np.mean(np.array(tri_errors) ** 2))
    print(f"      Triangulation RMSE = {tri_rmse:.4f} mm")

    print("\n  Step 6 complete.")


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Datalus calibration pipeline (steps 1-6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python DatulusCalib_Full.py           # full run\n"
            "  python DatulusCalib_Full.py --from 3  # resume from stereo calibration\n"
            "  python DatulusCalib_Full.py --from 4  # resume after filling CSV\n"
        )
    )
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1, metavar="N",
        help="Start from step N (1=frames, 2=intrinsics, 3=colmap, "
             "4=world_reg, 5=yaml, 6=validate) — 3=stereo replaces COLMAP"
    )
    args      = parser.parse_args()
    from_step = args.from_step

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    frames_root  = Path(OUTPUT_DIR) / "frames"
    intrinsics   = {}
    colmap_poses = {}
    cam_n_frames = {}
    world_poses  = {}

    # ── Step 1 ──────────────────────────────────────────────────────────────────
    if from_step <= 1:
        frames_root, _ = step1_extract_frames(SESSIONS, OUTPUT_DIR, FRAME_INTERVAL_MS)

    # ── Step 2 ──────────────────────────────────────────────────────────────────
    if from_step <= 2:
        intrinsics = step2_calibrate_intrinsics(
            frames_root, CAMERA_IDS, CHESSBOARD, SQUARE_SIZE_MM,
            FOCAL_LENGTH_MM, SENSOR_WIDTH_MM, OUTPUT_DIR
        )
    else:
        intrinsics = _load_intrinsics(OUTPUT_DIR, CAMERA_IDS)

    # ── Step 3 ──────────────────────────────────────────────────────────────────
    if from_step <= 3:
        colmap_poses, cam_n_frames = step3_stereo(
            frames_root, intrinsics, CAMERA_IDS, OUTPUT_DIR
        )
    elif from_step == 4:
        # Only needed as input to step 4
        colmap_poses, cam_n_frames = _load_colmap_poses(OUTPUT_DIR, CAMERA_IDS)

    # ── Step 4 ──────────────────────────────────────────────────────────────────
    if from_step <= 4:
        world_poses, _, _, _ = step4_world_registration(
            colmap_poses, OUTPUT_DIR, WORLD_CSV, CAMERA_IDS
        )
        if world_poses is None:
            print("\n[PAUSED] Fill in world_registration.csv then re-run with --from 4")
            sys.exit(0)
    else:
        world_poses = _load_world_poses(OUTPUT_DIR, CAMERA_IDS)

    # ── Step 5 ──────────────────────────────────────────────────────────────────
    if from_step <= 5:
        step5_export_yaml(world_poses, intrinsics, OUTPUT_DIR, CAMERA_IDS)

    # ── Step 6 ──────────────────────────────────────────────────────────────────
    step6_validate(intrinsics, world_poses, WORLD_CSV, cam_n_frames, CAMERA_IDS)

    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"Output: {OUTPUT_DIR}/cam_{{102,108,113,117}}.yaml")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()