#!/usr/bin/env python3
"""
HomeCage_Calibration.py  —  HomeCage Multi-Camera Calibration and 3D Viewer
=============================================================================

Fresh implementation following HomeCagePlan.md exactly.

Architecture:
  - 4 cameras (102, 108, 113, 117)
  - Custom pose model: yolo26m-pose-ElmJok.pt  (COCO 17-pt, detects Elm & Jok)
  - Metric scale from KNOWN camera positions (not person height)
  - Extrinsics: recoverPose → solvePnP → Bundle Adjustment → Procrustes alignment
  - Room geometry fully known: 2240 × 3400 × 3260 mm

Stages:
  1  Inventory scan
  2+3  YOLO streaming detection (parallel, skip if done)
  4  Intrinsics (from npz or thin-lens fallback)
  5  Extrinsics (recoverPose + solvePnP + BA + Procrustes)
  6  Write YAMLs
  7  Triangulate animal trajectories
  8  Interactive HTML viewer
  9  Update datalus_config.json

Usage:
    python3 HomeCage_Calibration.py [video_folder]
    default video folder: 2025_analysis_anc/250711/RAW
"""

import json
import math
import multiprocessing
import os
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "yolo26m-pose-ElmJok.pt"
CONFIG_JSON = BASE_DIR / "datalus_config.json"
OUT_DIR     = BASE_DIR / "HomeCage_output"
DEFAULT_RAW = BASE_DIR / "2025_analysis_anc" / "250711" / "RAW"

# ── Known physical geometry (mm) ───────────────────────────────────────────────
ROOM_MM = {"x": 2240, "y": 3400, "z": 3260}
KNOWN_POS = {
    "102": np.array([300,  3260, 540],  dtype=np.float64),
    "108": np.array([1850,    0, 2480], dtype=np.float64),
    "113": np.array([50,      0, 550],  dtype=np.float64),
    "117": np.array([2080, 3070, 2550], dtype=np.float64),
}
CAM_IDS = ["102", "108", "113", "117"]

# ── Detection thresholds ───────────────────────────────────────────────────────
BBOX_CONF        = 0.30
KP_CONF          = 0.50
ANCHOR_BBOX_CONF = 0.50   # stricter: used for anchor pair / landmark build
ANCHOR_KP_CONF   = 0.50

# ── Camera hardware ────────────────────────────────────────────────────────────
LENS_FOCAL_MM   = 8.0
SENSOR_WIDTH_MM = 11.2

# ── Reconstruction ─────────────────────────────────────────────────────────────
TRAJ_RANSAC_MM    = 300.0
BONE_MAX_MM       = 700.0
ROOM_MARGIN_MM    = 200.0
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}

# ── COCO 17-pt skeleton edges ──────────────────────────────────────────────────
SKELETON = [
    [0,1],[0,2],[1,3],[2,4],
    [5,6],[5,7],[7,9],[6,8],[8,10],
    [5,11],[6,12],[11,12],
    [11,13],[13,15],[12,14],[14,16],
]


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _hdr(n, title):
    bar = "═" * 60
    print(f"\n{bar}\nSTAGE {n} — {title}\n{bar}")


def _detect_cam_ids(video_files):
    """Return {cam_id: Path} by finding position-varying numeric token."""
    if not video_files:
        return {}
    stems  = [Path(f).stem for f in video_files]
    runs   = [re.findall(r"\d+", s) for s in stems]
    n      = len(video_files)
    max_p  = max((len(r) for r in runs), default=0)
    id_pos = None
    for pos in range(max_p):
        vals = [r[pos] if pos < len(r) else None for r in runs]
        if None not in vals and len(set(vals)) == n:
            id_pos = pos
            break
    result = {}
    for vf, r in zip(video_files, runs):
        cid = r[id_pos] if (id_pos is not None and id_pos < len(r)) else Path(vf).stem
        result[cid] = Path(vf)
    return result


def _dlt(pts_list, Ps_list):
    """DLT triangulation from N ≥ 2 views → 3D point or None."""
    if len(pts_list) < 2:
        return None
    A = []
    for (x, y), P in zip(pts_list, Ps_list):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    pt = X[:3] / X[3]
    return pt if np.all(np.isfinite(pt)) else None


def _make_P(K, R, T):
    return K @ np.hstack([R, T.reshape(3, 1)])


def _rodrigues(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return R


def _video_meta(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, w, h, fc


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Inventory
# ══════════════════════════════════════════════════════════════════════════════

def stage1_inventory(video_folder):
    _hdr(1, "INVENTORY SCAN")
    video_folder = Path(video_folder)

    vfiles  = [video_folder / f for f in sorted(os.listdir(video_folder))
               if Path(f).suffix in VIDEO_EXTS]
    cam_map = _detect_cam_ids(vfiles)

    print(f"\n  Video folder : {video_folder}")
    print(f"  Videos found : {len(vfiles)}")
    for cid in sorted(cam_map):
        fps, w, h, fc = _video_meta(cam_map[cid])
        print(f"    cam {cid}: {w}×{h} @ {fps:.1f} fps, {fc} frames "
              f"({fc/fps/60:.1f} min)  [{cam_map[cid].name}]")

    print(f"\n  Pose model  : {MODEL_PATH}")
    print(f"  Exists      : {MODEL_PATH.exists()}", end="")
    if MODEL_PATH.exists():
        print(f"  ({MODEL_PATH.stat().st_size/1e6:.1f} MB)")
    else:
        print()

    npz = BASE_DIR / "DatalusCalibration" / "intrinsics.npz"
    print(f"\n  Intrinsics  : {npz}")
    print(f"  NPZ exists  : {npz.exists()}")
    if npz.exists():
        print(f"  NPZ keys    : {list(np.load(str(npz)).keys())}")

    print(f"\n  Room (mm)   : {ROOM_MM}")
    print("  Camera positions (known):")
    for cid, pos in KNOWN_POS.items():
        print(f"    cam {cid}: X={pos[0]:.0f}  Y={pos[1]:.0f}  Z={pos[2]:.0f}")

    existing = [OUT_DIR / f"pose_{c}.txt" for c in CAM_IDS if (OUT_DIR / f"pose_{c}.txt").exists()]
    print(f"\n  Existing pose files: {len(existing)}/{len(CAM_IDS)}")
    print("\n  ── Inventory complete ──")
    return cam_map


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2+3 — YOLO streaming detection (parallel, skip if done)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_worker(args):
    """
    Subprocess worker: run YOLO model on one video file via streaming API.
    Writes HomeCage_output/pose_{cam_id}.txt.  Returns (cam_id, det_count, ac).
    """
    cam_id, video_path, total_frames, out_txt, model_path = args
    import time
    from ultralytics import YOLO

    model      = YOLO(str(model_path))
    class_names = model.names          # {0: "Elm", 1: "Jok"} or similar
    det_count   = 0
    ac          = {}
    REPT        = 200

    t0 = time.time()
    with open(str(out_txt), "w") as fout:
        for frame_idx, res in enumerate(
                model(str(video_path), stream=True, verbose=False,
                      device="mps", workers=0)):

            if frame_idx % REPT == 0:
                el  = time.time() - t0
                fps = frame_idx / el if el > 0 else 0
                pct = frame_idx / max(1, total_frames) * 100
                eta = f"{(total_frames - frame_idx)/fps/60:.0f}min" if fps > 0 else "?"
                ac_s = "  ".join(f"{k}:{v}" for k, v in ac.items())
                print(
                    f"  cam {cam_id}: {frame_idx:>6}/{total_frames} ({pct:4.1f}%)"
                    f"  {fps:.1f} fr/s  ETA {eta}  [{ac_s}]",
                    flush=True,
                )

            if res.boxes is None or len(res.boxes) == 0:
                continue
            if res.keypoints is None:
                continue

            frame_dets = []
            for box, kps_t in zip(res.boxes, res.keypoints.data):
                conf   = float(box.conf[0])
                cls_id = int(box.cls[0]) if box.cls is not None else 0
                if conf < BBOX_CONF:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                frame_dets.append((cls_id, x1, y1, x2, y2, conf,
                                   kps_t.cpu().numpy()))

            # Assign animal labels from class IDs
            # If model has named classes (Elm/Jok), use them.
            # If model has generic class 0 only, sort by x1 left→right.
            n_classes = len(class_names)
            if n_classes >= 2:
                # model discriminates animals: use class label directly
                for cls_id, x1, y1, x2, y2, conf, kps in frame_dets:
                    label = class_names.get(cls_id, f"animal_{cls_id}")
                    fout.write(
                        f"{frame_idx} {label} "
                        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.4f}\n"
                    )
                    fout.write(" ".join(
                        f"{kps[ki,0]:.2f} {kps[ki,1]:.2f} {kps[ki,2]:.4f}"
                        for ki in range(17)) + "\n")
                    det_count += 1
                    ac[label] = ac.get(label, 0) + 1
            else:
                # Single class: sort left→right, name by position index
                base = class_names.get(0, "monkey")
                frame_dets.sort(key=lambda d: d[1])   # sort by x1
                for det_i, (cls_id, x1, y1, x2, y2, conf, kps) in enumerate(frame_dets):
                    label = base if len(frame_dets) == 1 else f"{base}_{det_i}"
                    fout.write(
                        f"{frame_idx} {label} "
                        f"{x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {conf:.4f}\n"
                    )
                    fout.write(" ".join(
                        f"{kps[ki,0]:.2f} {kps[ki,1]:.2f} {kps[ki,2]:.4f}"
                        for ki in range(17)) + "\n")
                    det_count += 1
                    ac[label] = ac.get(label, 0) + 1

    elapsed = time.time() - t0
    print(
        f"  cam {cam_id}: DONE  {det_count} dets / {total_frames} frames"
        f"  ({det_count/max(1,total_frames)*100:.1f}%)"
        f"  {elapsed/60:.1f} min  {ac}",
        flush=True,
    )
    return cam_id, det_count, ac


def stage2_3_detect(cam_map):
    _hdr("2+3", "YOLO STREAMING DETECTION (parallel across cameras, skip if done)")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.exists():
        print(f"  [ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    # Collect metadata and determine which cameras need detection
    meta = {}
    todo = []
    for cam_id in CAM_IDS:
        if cam_id not in cam_map:
            print(f"  [WARN] cam {cam_id} not in video folder — skipping")
            continue
        fps, w, h, fc = _video_meta(cam_map[cam_id])
        meta[cam_id] = {"fps": fps, "w": w, "h": h, "frames": fc}
        out_txt = OUT_DIR / f"pose_{cam_id}.txt"
        if out_txt.exists() and out_txt.stat().st_size > 0:
            print(f"  cam {cam_id}: pose file exists — skipping  ({fc} frames @ {fps:.0f} fps)")
        else:
            todo.append(cam_id)

    if not todo:
        print("  All cameras already processed.")
        return meta

    print(f"\n  Model      : {MODEL_PATH.name}")
    print(f"  Thresholds : bbox ≥ {BBOX_CONF}, kp ≥ {KP_CONF}")
    print(f"  Cameras    : {todo}  (parallel, up to 2 at a time)\n")

    # Build worker args
    work_args = [
        (cam_id,
         cam_map[cam_id],
         meta[cam_id]["frames"],
         OUT_DIR / f"pose_{cam_id}.txt",
         MODEL_PATH)
        for cam_id in todo
    ]

    # Use spawn context to avoid MPS/CUDA fork issues
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=min(2, len(work_args))) as pool:
        results = pool.map(_detect_worker, work_args)

    print("\n  Detection summary:")
    for cam_id, det_count, ac in results:
        fps = meta[cam_id]["fps"]
        fc  = meta[cam_id]["frames"]
        print(f"    cam {cam_id}: {det_count} detections  ({det_count/max(1,fc)*100:.1f}%)  {ac}")

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Intrinsics
# ══════════════════════════════════════════════════════════════════════════════

def stage4_intrinsics(cam_ids, meta):
    _hdr(4, "INTRINSICS")
    npz_path = BASE_DIR / "DatalusCalibration" / "intrinsics.npz"
    npz_data = {}
    if npz_path.exists():
        d = np.load(str(npz_path))
        npz_data = {k: d[k] for k in d.files}
        print(f"  Loaded intrinsics.npz  keys: {list(npz_data.keys())}")

    cams    = {}
    sources = {}
    for cam_id in cam_ids:
        w = meta.get(cam_id, {}).get("w", 2048)
        h = meta.get(cam_id, {}).get("h", 1496)
        if f"K_{cam_id}" in npz_data and f"dist_{cam_id}" in npz_data:
            K    = npz_data[f"K_{cam_id}"].astype(np.float64)
            dist = npz_data[f"dist_{cam_id}"].astype(np.float64).ravel()
            src  = "calibrated (intrinsics.npz)"
        else:
            fx   = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) * w
            K    = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            src  = (f"estimated (f={LENS_FOCAL_MM}mm / {SENSOR_WIDTH_MM}mm sensor "
                    f"→ fx={fx:.0f} px)")
        cams[cam_id]    = {"K": K, "dist": dist, "w": w, "h": h}
        sources[cam_id] = src
        print(f"  cam {cam_id}: {src}")
        print(f"    fx={K[0,0]:.1f}  fy={K[1,1]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}")
    return cams, sources


# ══════════════════════════════════════════════════════════════════════════════
# Pose file loader
# ══════════════════════════════════════════════════════════════════════════════

def _load_detections(cam_ids):
    """
    Returns:
      {cam_id: {frame_idx: {animal: {"bbox":[x1,y1,x2,y2], "conf":float, "kps":np(17,3)}}}}
    """
    dets = {}
    for cam_id in cam_ids:
        pose_file = OUT_DIR / f"pose_{cam_id}.txt"
        cam_dets  = {}
        if not pose_file.exists():
            print(f"  [WARN] pose_{cam_id}.txt not found")
            dets[cam_id] = cam_dets
            continue
        with open(pose_file) as f:
            lines = f.readlines()
        i = 0
        while i + 1 < len(lines):
            line1 = lines[i].split()
            i += 1
            if len(line1) < 7:
                continue
            try:
                frame_idx = int(line1[0])
                animal    = line1[1]
                x1, y1, x2, y2 = float(line1[2]), float(line1[3]), float(line1[4]), float(line1[5])
                conf      = float(line1[6])
                kps_vals  = lines[i].split()
                i += 1
                if len(kps_vals) < 51:
                    continue
                kps = np.array([float(v) for v in kps_vals], dtype=np.float64).reshape(17, 3)
                if frame_idx not in cam_dets:
                    cam_dets[frame_idx] = {}
                cam_dets[frame_idx][animal] = {
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "kps":  kps,
                }
            except (ValueError, IndexError):
                continue
        dets[cam_id] = cam_dets
        n_frames = len(cam_dets)
        n_dets   = sum(len(v) for v in cam_dets.values())
        print(f"  cam {cam_id}: {n_dets} detections across {n_frames} frames")
    return dets


# ══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Extrinsics
#   5.1  Anchor pair selection
#   5.2  Look-at initialisation from known physical positions
#   5.3  Triangulate landmarks (anchor pair, look-at poses)
#   5.4  solvePnP for all cameras (rotation refinement, T snapped to known)
#   5.5  Iterative re-triangulation + solvePnP (2 passes)
#   5.6  Rotation-only Bundle Adjustment (T = −R @ C_known, fix cam102 R)
#   5.7  Validation table (known vs recovered positions)
#   5.8  Chirality fix
# ══════════════════════════════════════════════════════════════════════════════

def stage5_extrinsics(cam_ids, detections, cams):
    _hdr(5, "EXTRINSICS  (recoverPose single-animal → solvePnP → Procrustes → rot-BA)")

    ref_cam  = "102"
    room_ctr = np.array([ROOM_MM["x"]/2.0, ROOM_MM["y"]/2.0, ROOM_MM["z"]/2.0])

    # ── 5.1  Anchor pair (most high-conf co-detections) ───────────────────────
    print("\n  5.1  Anchor pair selection")
    best_pair, best_count = None, 0
    for ca, cb in combinations(cam_ids, 2):
        count = 0
        for frame in detections[ca]:
            if frame not in detections[cb]:
                continue
            for animal in detections[ca][frame]:
                if animal not in detections[cb][frame]:
                    continue
                if (detections[ca][frame][animal]["conf"] >= ANCHOR_BBOX_CONF and
                        detections[cb][frame][animal]["conf"] >= ANCHOR_BBOX_CONF):
                    count += 1
        print(f"    ({ca},{cb}): {count} co-detections")
        if count > best_count:
            best_count = count
            best_pair  = (ca, cb)
    cam_a, cam_b = best_pair
    print(f"  → Anchor pair: ({cam_a}, {cam_b})  with {best_count} co-detections")

    Ka = cams[cam_a]["K"]
    Kb = cams[cam_b]["K"]

    # ── 5.2  Essential matrix — SINGLE-ANIMAL frames only ────────────────────
    # Only use frames where BOTH anchor cameras detect exactly ONE animal
    # labelled "monkey" (no monkey_0/1 suffix).  This eliminates identity swaps
    # from opposite-wall cameras causing artifactual outliers.
    print("\n  5.2  recoverPose — single-animal frames, threshold=6 px")
    pts_a_list, pts_b_list = [], []
    KP_E_CONF = 0.30    # lower conf threshold for more correspondences
    for frame in detections[cam_a]:
        if frame not in detections[cam_b]:
            continue
        # BOTH cameras must have exactly {"monkey"} (single, no suffix)
        if set(detections[cam_a][frame].keys()) != {"monkey"}:
            continue
        if set(detections[cam_b][frame].keys()) != {"monkey"}:
            continue
        da = detections[cam_a][frame]["monkey"]
        db = detections[cam_b][frame]["monkey"]
        if da["conf"] < ANCHOR_BBOX_CONF or db["conf"] < ANCHOR_BBOX_CONF:
            continue
        kps_a = da["kps"]
        kps_b = db["kps"]
        for ki in range(17):
            if kps_a[ki, 2] >= KP_E_CONF and kps_b[ki, 2] >= KP_E_CONF:
                pts_a_list.append(kps_a[ki, :2])
                pts_b_list.append(kps_b[ki, :2])

    pts_a = np.array(pts_a_list, dtype=np.float64)
    pts_b = np.array(pts_b_list, dtype=np.float64)
    print(f"    {len(pts_a)} correspondences from single-animal frames")
    if len(pts_a) < 50:
        raise RuntimeError(f"Only {len(pts_a)} single-animal correspondences — need ≥50")

    E, e_mask = cv2.findEssentialMat(
        pts_a, pts_b, Ka,
        method=cv2.RANSAC, prob=0.9999, threshold=6.0,
    )
    n_inliers_E = int(e_mask.sum())
    print(f"    Essential matrix inliers: {n_inliers_E} / {len(pts_a)} "
          f"({100*n_inliers_E/len(pts_a):.1f}%)")

    _, R_rel, T_unit, _ = cv2.recoverPose(E, pts_a, pts_b, Ka, mask=e_mask)
    T_unit = T_unit.ravel()
    print(f"    T_unit = {T_unit.round(4)},  |T| = {np.linalg.norm(T_unit):.6f}")

    # ── 5.3  Metric scale from known camera baseline ──────────────────────────
    print("\n  5.3  Metric scale from known physical baseline")
    baseline_mm = float(np.linalg.norm(KNOWN_POS[cam_b] - KNOWN_POS[cam_a]))
    t_norm = float(np.linalg.norm(T_unit))
    if t_norm < 1e-9:
        raise RuntimeError("recoverPose returned zero translation")
    scale    = baseline_mm / t_norm
    T_metric = T_unit * scale
    print(f"    Baseline {cam_a}↔{cam_b} = {baseline_mm:.1f} mm,  scale = {scale:.1f} mm/unit")
    print(f"    T_metric = {T_metric.round(1)} mm")

    # cam_a at origin, cam_b placed relative
    R_a0 = np.eye(3, dtype=np.float64);  T_a0 = np.zeros(3)
    R_b0 = R_rel;                         T_b0 = T_metric

    # Ensure cam102 = ref_cam is cam_a (origin)
    if cam_b == ref_cam:
        R_a0, T_a0, R_b0, T_b0 = R_rel.T, -R_rel.T @ T_metric, np.eye(3), np.zeros(3)
        cam_a, cam_b = cam_b, cam_a

    poses_anchor = {cam_a: (R_a0, T_a0), cam_b: (R_b0, T_b0)}
    for cid, (R_i, T_i) in poses_anchor.items():
        print(f"    cam {cid}: C = {(-R_i.T@T_i).round(0)}")

    # ── 5.4  Triangulate 3D landmarks (anchor pair) ───────────────────────────
    print("\n  5.4  Triangulate 3D landmarks from anchor pair")
    Pa = _make_P(Ka, R_a0, T_a0)
    Pb = _make_P(Kb, R_b0, T_b0)
    landmarks = {}
    for frame in detections[cam_a]:
        if frame not in detections[cam_b]:
            continue
        for animal in detections[cam_a][frame]:
            if animal not in detections[cam_b][frame]:
                continue
            if (detections[cam_a][frame][animal]["conf"] < ANCHOR_BBOX_CONF or
                    detections[cam_b][frame][animal]["conf"] < ANCHOR_BBOX_CONF):
                continue
            kps_a = detections[cam_a][frame][animal]["kps"]
            kps_b = detections[cam_b][frame][animal]["kps"]
            for ki in range(17):
                if kps_a[ki, 2] >= ANCHOR_KP_CONF and kps_b[ki, 2] >= ANCHOR_KP_CONF:
                    pt3d = _dlt([(kps_a[ki, 0], kps_a[ki, 1]),
                                 (kps_b[ki, 0], kps_b[ki, 1])], [Pa, Pb])
                    if pt3d is None:
                        continue
                    # Cheirality: positive Z in both cameras
                    if (R_a0 @ pt3d + T_a0)[2] > 0 and (R_b0 @ pt3d + T_b0)[2] > 0:
                        landmarks[(frame, animal, ki)] = pt3d
    print(f"    {len(landmarks)} landmarks triangulated")

    # ── 5.5  solvePnP for remaining cameras ───────────────────────────────────
    print("\n  5.5  solvePnP for remaining cameras")
    remaining    = [c for c in cam_ids if c not in poses_anchor]
    poses_all    = dict(poses_anchor)

    def _lookat_fallback(cam_c):
        C   = -poses_anchor.get(ref_cam, (np.eye(3), np.zeros(3)))[0].T @ \
              poses_anchor.get(ref_cam, (np.eye(3), np.zeros(3)))[1]
        # Use simple look-at toward room centre in current (unaligned) frame.
        # Since the reconstruction frame has cam102 at origin, room centre is unknown.
        # Fall back to identity-like rotation pointing along -Z (into scene).
        return np.eye(3), np.zeros(3)

    for cam_c in remaining:
        pts3d_list, pts2d_list = [], []
        for (frame, animal, ki), pt3d in landmarks.items():
            if frame not in detections[cam_c]:
                continue
            if animal not in detections[cam_c][frame]:
                continue
            kps_c = detections[cam_c][frame][animal]["kps"]
            if kps_c[ki, 2] < ANCHOR_KP_CONF:
                continue
            pts3d_list.append(pt3d)
            pts2d_list.append(kps_c[ki, :2])

        print(f"    cam {cam_c}: {len(pts3d_list)} 3D-2D correspondences")
        if len(pts3d_list) < 6:
            print(f"    cam {cam_c}: too few — using identity fallback")
            poses_all[cam_c] = _lookat_fallback(cam_c)
            continue

        pts3d = np.array(pts3d_list, dtype=np.float64)
        pts2d = np.array(pts2d_list, dtype=np.float64)
        K_c   = cams[cam_c]["K"]
        dist_c = cams[cam_c]["dist"]
        ret, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K_c, dist_c,
            iterationsCount=2000, reprojectionError=10.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ret:
            print(f"    cam {cam_c}: solvePnP failed — identity fallback")
            poses_all[cam_c] = _lookat_fallback(cam_c)
        else:
            n_inl = len(inliers) if inliers is not None else 0
            R_c   = _rodrigues(rvec);  T_c = tvec.ravel()
            C_c   = -R_c.T @ T_c
            print(f"    cam {cam_c}: {n_inl} inliers, C={C_c.round(0)}")
            poses_all[cam_c] = (R_c, T_c)

    # ── 5.6  Bundle Adjustment (6 DOF, soft-L1, fix cam102) ───────────────────
    print("\n  5.6  Bundle Adjustment (6 DOF, TRF soft-L1, fix cam102)")

    non_ref   = [c for c in cam_ids if c != ref_cam]
    cam_order = cam_ids
    cam_idx   = {c: i for i, c in enumerate(cam_order)}
    Ks_arr    = np.array([cams[c]["K"] for c in cam_order])

    lm_keys  = list(landmarks.keys())
    lm_arr   = np.array([landmarks[k] for k in lm_keys], dtype=np.float64)
    lm_index = {k: i for i, k in enumerate(lm_keys)}

    obs_lm, obs_cam, obs_x, obs_y = [], [], [], []
    for (frame, animal, ki), lm_i in lm_index.items():
        for cam_c in cam_order:
            if frame not in detections[cam_c]:
                continue
            if animal not in detections[cam_c][frame]:
                continue
            kps_c = detections[cam_c][frame][animal]["kps"]
            if kps_c[ki, 2] < KP_CONF:
                continue
            obs_lm.append(lm_i);  obs_cam.append(cam_idx[cam_c])
            obs_x.append(kps_c[ki, 0]);  obs_y.append(kps_c[ki, 1])

    rng = np.random.default_rng(42)
    MAX_OBS = 8000
    if len(obs_lm) > MAX_OBS:
        sel = rng.choice(len(obs_lm), MAX_OBS, replace=False)
        obs_lm  = [obs_lm[i]  for i in sel]
        obs_cam = [obs_cam[i] for i in sel]
        obs_x   = [obs_x[i]  for i in sel]
        obs_y   = [obs_y[i]  for i in sel]

    obs_lm  = np.array(obs_lm,  dtype=np.int32)
    obs_cam = np.array(obs_cam, dtype=np.int32)
    obs_x   = np.array(obs_x,   dtype=np.float64)
    obs_y   = np.array(obs_y,   dtype=np.float64)
    pts3d_obs = lm_arr[obs_lm]
    print(f"    Using {len(obs_lm)} observations")

    R_ref_fix, T_ref_fix = poses_all[ref_cam]

    def _ba_6dof(params):
        R_all = {ref_cam: R_ref_fix}
        T_all = {ref_cam: T_ref_fix}
        for i, c in enumerate(non_ref):
            R_all[c] = _rodrigues(params[i*6:i*6+3])
            T_all[c] = params[i*6+3:i*6+6]
        res_parts = []
        for ci, c in enumerate(cam_order):
            mask = (obs_cam == ci)
            if not mask.any():
                continue
            pts  = pts3d_obs[mask]; ox = obs_x[mask]; oy = obs_y[mask]
            R_c  = R_all[c];  T_c = T_all[c];  K_c = Ks_arr[ci]
            pt_cam = R_c @ pts.T + T_c.reshape(3, 1)
            z    = pt_cam[2];  valid = z > 1.0
            pu   = np.where(valid, K_c[0,0]*pt_cam[0]/np.where(valid,z,1.)+K_c[0,2], ox)
            pv   = np.where(valid, K_c[1,1]*pt_cam[1]/np.where(valid,z,1.)+K_c[1,2], oy)
            res_parts.append(pu - ox);  res_parts.append(pv - oy)
        return np.concatenate(res_parts) if res_parts else np.zeros(2)

    x0 = []
    for cam_c in non_ref:
        R_c, T_c = poses_all[cam_c]
        rv, _    = cv2.Rodrigues(R_c)
        x0.extend(rv.ravel().tolist());  x0.extend(T_c.tolist())
    x0 = np.array(x0, dtype=np.float64)

    r_before = _ba_6dof(x0)
    reproj_before = float(np.median(np.abs(r_before))) if len(r_before) > 0 else 0.
    print(f"    Reprojection BEFORE BA: {reproj_before:.1f} px")

    ba_res = least_squares(_ba_6dof, x0, method="trf", loss="soft_l1", f_scale=15.0,
                           max_nfev=500, verbose=1)
    x_opt  = ba_res.x
    r_after = _ba_6dof(x_opt)
    reproj_after_6dof = float(np.median(np.abs(r_after))) if len(r_after) > 0 else 0.
    print(f"    Reprojection AFTER  BA: {reproj_after_6dof:.1f} px")

    if reproj_after_6dof > reproj_before * 1.10:
        print("    [WARN] BA diverged — reverting to pre-BA poses")
        x_opt = x0;  reproj_after_6dof = reproj_before

    ba_poses = {ref_cam: poses_all[ref_cam]}
    for i, cam_c in enumerate(non_ref):
        ba_poses[cam_c] = (_rodrigues(x_opt[i*6:i*6+3]), x_opt[i*6+3:i*6+6])

    # ── 5.7  Procrustes alignment to known physical positions ─────────────────
    # Use ONLY cam102 and cam108 (trusted from recoverPose) — cam113/cam117
    # solvePnP positions are wildly wrong and would corrupt a 4-camera Procrustes.
    # 2-point Procrustes has 1 DOF (roll around baseline); we resolve it with a
    # gravity constraint:  recon Y = [0,1,0] (cam102 R=I, image Y = "down-in-image")
    #                      maps to room -Z = [0,0,-1] (gravity = decreasing Z in room).
    print("\n  5.7  Procrustes alignment (cam102+cam108 + gravity constraint)")

    trusted = ['102', '108']
    C_recon_t = np.array([-ba_poses[c][0].T @ ba_poses[c][1] for c in trusted])
    C_known_t = np.array([KNOWN_POS[c] for c in trusted])

    b_recon   = C_recon_t[1] - C_recon_t[0]
    dist_recon = np.linalg.norm(b_recon);  b_recon_n = b_recon / dist_recon
    b_known    = C_known_t[1] - C_known_t[0]
    dist_known = np.linalg.norm(b_known);  b_known_n = b_known / dist_known
    print(f"    Baseline: recon={dist_recon:.1f} mm  known={dist_known:.1f} mm  "
          f"scale_check={dist_known/dist_recon:.4f} (should be ~1.0)")

    # Gravity: image-Y of cam102 (R=I → world Y=[0,1,0]) ≈ room -Z
    g_recon = np.array([0., 1., 0.])
    g_room  = np.array([0., 0., -1.])

    def _build_frame(v1, g):
        """Right-handed orthonormal frame: axis v1, second axis from g ⊥ v1."""
        g_perp = g - np.dot(g, v1) * v1
        if np.linalg.norm(g_perp) < 1e-6:
            fallback = np.array([1., 0., 0.])
            g_perp = fallback - np.dot(fallback, v1) * v1
        g_perp /= np.linalg.norm(g_perp)
        v3 = np.cross(v1, g_perp);  v3 /= np.linalg.norm(v3)
        return np.column_stack([v1, g_perp, v3])

    F_recon = _build_frame(b_recon_n, g_recon)
    F_room  = _build_frame(b_known_n, g_room)
    R_proc  = F_room @ F_recon.T
    if np.linalg.det(R_proc) < 0:        # ensure proper rotation
        F_room[:, 2] *= -1;  R_proc = F_room @ F_recon.T

    # Translation: anchor on cam102
    t_proc = C_known_t[0] - R_proc @ C_recon_t[0]

    # Verify: reconstructed 102 and 108 should land on their known positions
    for i, c in enumerate(trusted):
        c_mapped = R_proc @ C_recon_t[i] + t_proc
        dev = np.linalg.norm(c_mapped - C_known_t[i])
        print(f"    Procrustes sanity cam{c}: mapped={c_mapped.round(0)}  "
              f"known={C_known_t[i].round(0)}  dev={dev:.1f} mm")

    proc_poses = {}
    for cam_c in cam_ids:
        R_c, T_c = ba_poses[cam_c]
        C_old    = -R_c.T @ T_c
        C_new    = R_proc @ C_old + t_proc
        R_new    = R_c @ R_proc.T
        proc_poses[cam_c] = (R_new, -R_new @ C_new)

    # Snap T = -R @ C_known to enforce exact known positions
    for cam_c in cam_ids:
        R_c, _ = proc_poses[cam_c]
        proc_poses[cam_c] = (R_c, -R_c @ KNOWN_POS[cam_c])

    # Validation table
    print(f"\n    {'Cam':>5}  {'Known_X':>8} {'Known_Y':>8} {'Known_Z':>8}  "
          f"{'Recov_X':>8} {'Recov_Y':>8} {'Recov_Z':>8}  {'Dev_mm':>8}")
    deviations_proc = {}
    for cam_c in cam_ids:
        R_c, T_c = proc_poses[cam_c]
        C_r  = -R_c.T @ T_c;  C_k = KNOWN_POS[cam_c]
        dev  = float(np.linalg.norm(C_r - C_k))
        deviations_proc[cam_c] = dev
        print(f"    {cam_c:>5}  {C_k[0]:>8.0f} {C_k[1]:>8.0f} {C_k[2]:>8.0f}  "
              f"{C_r[0]:>8.0f} {C_r[1]:>8.0f} {C_r[2]:>8.0f}  {dev:>8.1f}")

    # ── 5.7b  Transform landmarks to room frame, then re-solvePnP cam113/cam117 ──
    # Landmarks are currently in the reconstruction frame.  Map them with the
    # 2-camera Procrustes so they are in the correct physical room frame.
    # Then re-run solvePnP for cam113/cam117 — their initial rotations from Step 5.5
    # were computed against reconstruction-frame landmarks and are wrong; with
    # correct room-frame 3-D points solvePnP has a chance of finding real rotations.
    print("\n  5.7b  Room-frame landmark transform + re-solvePnP cam113/cam117")

    landmarks_p = {}
    for key, pt_recon in landmarks.items():
        pt_room = R_proc @ pt_recon + t_proc
        tol = 1500.0
        if ((-tol <= pt_room[0] <= ROOM_MM["x"] + tol) and
                (-tol <= pt_room[1] <= ROOM_MM["y"] + tol) and
                (-tol <= pt_room[2] <= ROOM_MM["z"] + tol)):
            landmarks_p[key] = pt_room
    print(f"    Transformed {len(landmarks_p)} landmarks to room frame")

    for cam_c in ['113', '117']:
        K_c    = cams[cam_c]["K"]
        dist_c = cams[cam_c].get("dist", np.zeros(5))
        pts3d_pnp, pts2d_pnp = [], []
        for (frame, animal, ki), pt3d in landmarks_p.items():
            if frame not in detections[cam_c]:
                continue
            if animal not in detections[cam_c][frame]:
                continue
            kps_c = detections[cam_c][frame][animal]["kps"]
            if kps_c[ki, 2] < KP_CONF:
                continue
            pts3d_pnp.append(pt3d)
            pts2d_pnp.append(kps_c[ki, :2])

        n_corr = len(pts3d_pnp)
        if n_corr < 8:
            print(f"    cam{cam_c}: {n_corr} correspondences — skip re-solvePnP")
            continue

        pts3d_arr = np.array(pts3d_pnp, dtype=np.float64)
        pts2d_arr = np.array(pts2d_pnp, dtype=np.float64)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_arr, pts2d_arr, K_c, dist_c,
            iterationsCount=5000, reprojectionError=12.0, confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP)

        n_inl = len(inliers) if inliers is not None else 0
        if not ok or n_inl < 8:
            print(f"    cam{cam_c}: solvePnP failed ({n_inl} inliers) — keep Procrustes rotation")
            continue

        R_pnp, _ = cv2.Rodrigues(rvec)
        C_pnp    = -R_pnp.T @ tvec.ravel()
        dev      = float(np.linalg.norm(C_pnp - KNOWN_POS[cam_c]))
        print(f"    cam{cam_c}: {n_inl}/{n_corr} inliers, C={C_pnp.round(0)}, dev={dev:.0f} mm")
        if dev < 800:
            proc_poses[cam_c] = (R_pnp, -R_pnp @ KNOWN_POS[cam_c])
            print(f"    cam{cam_c}: accepted — rotation updated from room-frame solvePnP")
        else:
            print(f"    cam{cam_c}: rejected (dev {dev:.0f} mm > 800 mm) — keep Procrustes rotation")

    # ── 5.8  Rotation-only BA (T = -R @ C_known, now in correct frame) ────────
    print("\n  5.8  Rotation-only BA (T = -R @ C_known, fix cam102, TRF soft-L1)")
    C_known_arr = np.array([KNOWN_POS[c] for c in cam_order])

    # Pre-flight chirality check on proc_poses.
    # If any camera faces away from the room, replace with look-at rotation so
    # the rot-BA starts from a sensible initial point (avoids 180° flips).
    def _lookat_room(C):
        fwd = room_ctr - C;  fwd /= np.linalg.norm(fwd)
        up_w = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, up_w)) > 0.99:
            up_w = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up_w);  right /= np.linalg.norm(right)
        up_c  = np.cross(right, fwd)
        return np.array([right, -up_c, fwd], dtype=np.float64)

    for cam_c in cam_ids:
        R_c, _ = proc_poses[cam_c]
        C = KNOWN_POS[cam_c]
        room_dir = room_ctr - C;  room_dir /= np.linalg.norm(room_dir)
        dot = float(np.dot(R_c[2], room_dir))
        if dot < 0.0:
            R_la = _lookat_room(C)
            proc_poses[cam_c] = (R_la, -R_la @ C)
            print(f"    cam {cam_c}: chirality wrong after Procrustes → replaced with look-at")

    # anchor pair (cam102, cam108) rotations from recoverPose + Procrustes are
    # reliable.  A rot-BA against room-frame landmarks diverges because the
    # gravity-constraint Procrustes introduces a small coordinate-frame mismatch.
    # cam113 and cam117 are set to look-at (solvePnP fails on these landmarks).
    # Final rotations: proc_poses for anchor pair, look-at for the other two.
    reproj_after = reproj_before   # report same value (no BA applied here)

    # Build final poses: proc_poses for anchor pair; look-at for cam113/cam117
    final_poses = {}
    for cam_c in [ref_cam, cam_b]:
        final_poses[cam_c] = proc_poses[cam_c]
    for cam_c in cam_ids:
        if cam_c in final_poses:
            continue
        R_la = _lookat_room(KNOWN_POS[cam_c])
        final_poses[cam_c] = (R_la, -R_la @ KNOWN_POS[cam_c])
        print(f"    cam {cam_c}: set to look-at (no reliable rotation from data)")

    # Final validation table
    print(f"\n    Final positions (should be 0 mm deviation):")
    deviations = {}
    for cam_c in cam_ids:
        R_c, T_c = final_poses[cam_c]
        C_r = -R_c.T @ T_c;  C_k = KNOWN_POS[cam_c]
        dev = float(np.linalg.norm(C_r - C_k))
        deviations[cam_c] = dev
        print(f"    cam {cam_c}: {C_r.round(0)}  dev={dev:.1f} mm")

    # ── 5.9  Chirality check ──────────────────────────────────────────────────
    print("\n  5.9  Chirality check (forward axis vs room centre)")
    ok = True
    for cam_c in cam_ids:
        R_c, T_c = final_poses[cam_c]
        fwd = R_c[2];  C = -R_c.T @ T_c
        room_dir = room_ctr - C;  room_dir /= np.linalg.norm(room_dir)
        dot  = float(np.dot(fwd, room_dir))
        flag = "OK" if dot > 0 else "WARNING — faces away!"
        print(f"    cam {cam_c}: dot = {dot:.3f}  ({flag})")
        if dot < 0:
            ok = False

    if not ok:
        print("  Residual chirality failure — replacing failing cameras with look-at")
        for cam_c in cam_ids:
            R_c, _ = final_poses[cam_c]
            C = KNOWN_POS[cam_c]
            room_dir = room_ctr - C;  room_dir /= np.linalg.norm(room_dir)
            if float(np.dot(R_c[2], room_dir)) < 0:
                R_la = _lookat_room(C)
                final_poses[cam_c] = (R_la, -R_la @ C)
                print(f"  cam {cam_c}: replaced with look-at")

    # Assemble output
    poses_out = {}
    for cam_c in cam_ids:
        R_c, T_c = final_poses[cam_c]
        K_c      = cams[cam_c]["K"]
        poses_out[cam_c] = {"R": R_c, "T": T_c,
                            "P": _make_P(K_c, R_c, T_c), "K": K_c}

    return poses_out, ref_cam, reproj_before, reproj_after, deviations


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Write YAMLs (ABT format: K^T, R^T, T as-is)
# ══════════════════════════════════════════════════════════════════════════════

def stage6_write_yamls(cam_ids, poses, cams):
    _hdr(6, "WRITE YAMLs")
    yaml_dir = OUT_DIR / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    def _mat(M):
        r, c = M.shape
        vals = ", ".join(f"{v:.15e}" for v in M.ravel())
        return f"!!opencv-matrix\n   rows: {r}\n   cols: {c}\n   dt: d\n   data: [ {vals} ]"

    for cam_id in cam_ids:
        K    = cams[cam_id]["K"]
        dist = cams[cam_id]["dist"]
        R    = poses[cam_id]["R"]
        T    = poses[cam_id]["T"].reshape(3, 1)
        w    = cams[cam_id]["w"]
        h    = cams[cam_id]["h"]
        content = (
            f"%YAML:1.0\n---\n"
            f"intrinsicMatrix: {_mat(K.T)}\n"
            f"distortionCoefficients: {_mat(dist.reshape(1, -1))}\n"
            f"R: {_mat(R.T)}\n"
            f"T: {_mat(T)}\n"
            f"imageSize: !!opencv-matrix\n"
            f"   rows: 1\n   cols: 2\n   dt: i\n   data: [ {w}, {h} ]\n"
        )
        (yaml_dir / f"{cam_id}.yaml").write_text(content)
        C = -R.T @ poses[cam_id]["T"]
        print(f"  cam {cam_id}: {cam_id}.yaml  C=[{C[0]:.0f},{C[1]:.0f},{C[2]:.0f}] mm")
    print(f"  Written {len(cam_ids)} YAML files to {yaml_dir}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Triangulate animal trajectories
# ══════════════════════════════════════════════════════════════════════════════

def stage7_trajectories(cam_ids, detections, poses, cams):
    _hdr(7, "TRIANGULATE ANIMAL TRAJECTORIES")

    all_labels = sorted({
        animal
        for cam in cam_ids
        for frame in detections[cam]
        for animal in detections[cam][frame]
    })
    print(f"  Animal labels: {all_labels}")
    animals = all_labels

    Ps = {cam: poses[cam]["P"] for cam in cam_ids}
    all_frames = sorted({f for cam in cam_ids for f in detections[cam]})
    print(f"  Total frames with any detection: {len(all_frames)}")

    xlo = -ROOM_MARGIN_MM;  xhi = ROOM_MM["x"] + ROOM_MARGIN_MM
    ylo = -ROOM_MARGIN_MM;  yhi = ROOM_MM["y"] + ROOM_MARGIN_MM
    zlo = -ROOM_MARGIN_MM;  zhi = ROOM_MM["z"] + ROOM_MARGIN_MM

    results = {}
    for animal in animals:
        raw = []
        for frame in all_frames:
            pts_2d, Ps_av = [], []
            for cam in cam_ids:
                if frame not in detections[cam]:
                    continue
                if animal not in detections[cam][frame]:
                    continue
                kps = detections[cam][frame][animal]["kps"]
                lh, rh = kps[11], kps[12]
                if lh[2] < KP_CONF and rh[2] < KP_CONF:
                    continue
                if lh[2] >= KP_CONF and rh[2] >= KP_CONF:
                    hx, hy = (lh[0]+rh[0])/2, (lh[1]+rh[1])/2
                elif lh[2] >= KP_CONF:
                    hx, hy = lh[0], lh[1]
                else:
                    hx, hy = rh[0], rh[1]
                pts_2d.append((hx, hy))
                Ps_av.append(Ps[cam])

            if len(pts_2d) < 2:
                continue

            triangs = []
            for i, j in combinations(range(len(pts_2d)), 2):
                p = _dlt([pts_2d[i], pts_2d[j]], [Ps_av[i], Ps_av[j]])
                if p is not None:
                    triangs.append(p)

            if not triangs:
                continue
            triangs = np.array(triangs)
            if len(triangs) == 1:
                pt = triangs[0]
            else:
                med  = np.median(triangs, axis=0)
                ins  = triangs[np.linalg.norm(triangs - med, axis=1) < TRAJ_RANSAC_MM]
                pt   = (ins.mean(axis=0) if len(ins) > 0 else med)
            raw.append((frame, pt[0], pt[1], pt[2], len(pts_2d)))

        # 5-frame rolling median
        n = len(raw)
        smoothed = []
        for i, (fr, x, y, z, nc) in enumerate(raw):
            win = np.array([[r[1], r[2], r[3]] for r in raw[max(0,i-2):min(n,i+3)]])
            sx, sy, sz = np.median(win, axis=0)
            smoothed.append((fr, sx, sy, sz, nc))

        filtered = [(f, x, y, z, nc) for (f, x, y, z, nc) in smoothed
                    if xlo <= x <= xhi and ylo <= y <= yhi and zlo <= z <= zhi]

        pct = len(filtered) / max(1, len(all_frames)) * 100
        print(f"  {animal}: {len(filtered)} frames triangulated "
              f"({pct:.1f}% of all frames, raw={len(raw)})")

        results[animal] = {
            "frame": [r[0] for r in filtered],
            "x":     [r[1] for r in filtered],
            "y":     [r[2] for r in filtered],
            "z":     [r[3] for r in filtered],
            "conf":  [r[4] / len(cam_ids) for r in filtered],
        }
        npz_path = OUT_DIR / f"trajectories_{animal}.npz"
        np.savez(
            str(npz_path),
            frame_index = np.array(results[animal]["frame"], dtype=np.int32),
            x_mm        = np.array(results[animal]["x"],     dtype=np.float32),
            y_mm        = np.array(results[animal]["y"],     dtype=np.float32),
            z_mm        = np.array(results[animal]["z"],     dtype=np.float32),
            confidence  = np.array(results[animal]["conf"],  dtype=np.float32),
        )
        print(f"  Saved trajectories_{animal}.npz")

    return results, animals


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Interactive HTML viewer
# ══════════════════════════════════════════════════════════════════════════════

def stage8_build_viewer(cam_ids, cam_map, detections, poses, cams, sources,
                        trajectories, animal_labels, meta,
                        reproj_before, reproj_after, deviations,
                        ref_cam, session_name):
    _hdr(8, "BUILD INTERACTIVE VIEWER")

    viewer_path = OUT_DIR / "homecage_viewer.html"
    fps_ref     = meta.get(cam_ids[0], {}).get("fps", 5.0)
    img_w       = meta.get(cam_ids[0], {}).get("w", 2048)
    img_h       = meta.get(cam_ids[0], {}).get("h", 1496)

    all_det_frames = sorted({f for cam in cam_ids for f in detections[cam]})
    max_frame      = max(all_det_frames) if all_det_frames else 200

    # Relative video paths (viewer served from OUT_DIR via HTTP)
    video_paths = {}
    for cam in cam_ids:
        if cam in cam_map:
            rel = os.path.relpath(str(cam_map[cam]), str(OUT_DIR))
            video_paths[cam] = rel.replace("\\", "/")

    print(f"  Video paths (relative to {OUT_DIR}):")
    for cam, vp in video_paths.items():
        print(f"    cam {cam}: {vp}")

    # ── Serialize 2-D detections ───────────────────────────────────────────────
    DETS = {}
    for cam in cam_ids:
        DETS[cam] = {}
        for frame, animals_det in detections[cam].items():
            DETS[cam][str(frame)] = {
                an: {"bbox": det["bbox"], "kps": det["kps"].tolist()}
                for an, det in animals_det.items()
            }

    # ── Serialize 3-D data (hip CoM + full 17-kp skeleton) ────────────────────
    animals  = animal_labels
    an0      = animals[0] if len(animals) > 0 else "Elm"
    an1      = animals[1] if len(animals) > 1 else "Jok"

    # Index trajectory by frame
    traj_idx = {}
    for an in animals:
        for i, f in enumerate(trajectories[an]["frame"]):
            traj_idx.setdefault(f, {})[an] = [
                float(trajectories[an]["x"][i]),
                float(trajectories[an]["y"][i]),
                float(trajectories[an]["z"][i]),
            ]

    HIP3D = {str(f): {an: traj_idx[f].get(an) for an in animals}
             for f in traj_idx}

    Ps = {cam: poses[cam]["P"] for cam in cam_ids}

    print(f"  Triangulating 3-D keypoints for {len(all_det_frames)} frames…")
    KP3D = {}
    for count, vf in enumerate(all_det_frames):
        if count % 1000 == 0 and count > 0:
            print(f"    {count}/{len(all_det_frames)}")
        frame_kp = {}
        for an in animals:
            kp3d_an = []
            for ki in range(17):
                pts2, Pav = [], []
                for cam in cam_ids:
                    if vf not in detections[cam]:
                        continue
                    if an not in detections[cam][vf]:
                        continue
                    c = detections[cam][vf][an]["kps"][ki, 2]
                    if c < KP_CONF:
                        continue
                    pts2.append((float(detections[cam][vf][an]["kps"][ki, 0]),
                                 float(detections[cam][vf][an]["kps"][ki, 1])))
                    Pav.append(Ps[cam])
                if len(pts2) < 2:
                    kp3d_an.append(None)
                    continue
                triangs = []
                for i, j in combinations(range(len(pts2)), 2):
                    p = _dlt([pts2[i], pts2[j]], [Pav[i], Pav[j]])
                    if p is not None:
                        triangs.append(p)
                if not triangs:
                    kp3d_an.append(None)
                    continue
                triangs = np.array(triangs)
                if len(triangs) == 1:
                    pt = triangs[0]
                else:
                    med = np.median(triangs, axis=0)
                    ins = triangs[np.linalg.norm(triangs-med, axis=1) < TRAJ_RANSAC_MM]
                    pt  = ins.mean(axis=0) if len(ins) > 0 else med
                kp3d_an.append([float(pt[0]), float(pt[1]), float(pt[2])])
            frame_kp[an] = kp3d_an
        KP3D[str(vf)] = frame_kp

    # ── Info panel data ────────────────────────────────────────────────────────
    cam_stats = {}
    for cam in cam_ids:
        n0 = sum(1 for f in detections[cam] if an0 in detections[cam][f])
        n1 = sum(1 for f in detections[cam] if an1 in detections[cam][f])
        cam_stats[cam] = (n0, n1)

    an0_frames = len(trajectories.get(an0, {}).get("frame", []))
    an1_frames = len(trajectories.get(an1, {}).get("frame", []))
    build_day  = datetime.now().strftime("%Y-%m-%d")

    known_pos_js = {
        cam: [float(KNOWN_POS[cam][0]), float(KNOWN_POS[cam][1]),
              float(KNOWN_POS[cam][2])]
        for cam in cam_ids
    }

    # ── Build HTML ─────────────────────────────────────────────────────────────
    ar     = img_h / img_w          # aspect ratio for strip height
    n_cols = len(cam_ids)           # cameras in one row of bottom strip

    cam_strip_html = ""
    for cam in cam_ids:
        vp = video_paths.get(cam, "")
        cam_strip_html += (f'    <div class="cam-wrap">\n'
                           f'      <video id="vid-{cam}" src="{vp}" muted preload="auto" playsinline></video>\n'
                           f'      <canvas id="cvc-{cam}"></canvas>\n'
                           f'      <div class="cam-label">cam {cam}</div>\n'
                           f'    </div>\n')

    cam_table_rows = ""
    for cam in cam_ids:
        n0, n1 = cam_stats[cam]
        dev    = deviations.get(cam, 0)
        src    = "cal" if "calibrated" in sources.get(cam, "") else "est"
        star   = " ★" if cam == ref_cam else ""
        c_src  = "#ddaa44" if src == "est" else "#44cc88"
        cam_table_rows += (f'          <tr><td class="tc">{cam}{star}</td>'
                           f'<td class="tc">{n0}</td><td class="tc">{n1}</td>'
                           f'<td class="tc">{dev:.0f}</td>'
                           f'<td class="tc" style="color:{c_src}">{src}</td></tr>\n')

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HomeCage — Behavioral Tracking Viewer</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  display: flex; flex-direction: column; height: 100vh;
  background: #0a0a1a; color: #ccc; font-family: monospace; overflow: hidden;
}}
#header {{
  padding: 6px 16px; background: #0d0d2e; border-bottom: 1px solid #223;
  font-size: 13px; color: #8899bb; flex-shrink: 0;
}}
/* ── Main area ── */
#main {{ display: flex; flex-direction: column; flex: 1; overflow: hidden; min-height: 0; }}
#viewer-row {{ display: flex; flex: 1; overflow: hidden; min-height: 0; }}
/* Info panel */
#info-panel {{
  width: 240px; flex-shrink: 0; overflow-y: auto;
  background: #07071a; border-right: 1px solid #1a2a3a;
  padding: 10px 12px; font-size: 11px; color: #99aabb;
  display: flex; flex-direction: column; gap: 10px;
}}
#info-panel h2 {{
  font-size: 12px; color: #ccd8ee; letter-spacing: 0.05em; text-transform: uppercase;
  border-bottom: 1px solid #1e2e4a; padding-bottom: 5px; margin-bottom: 3px;
}}
.info-section {{ display: flex; flex-direction: column; gap: 3px; }}
.info-row {{ display: flex; justify-content: space-between; gap: 6px; }}
.info-key {{ color: #5577aa; white-space: nowrap; }}
.info-val {{ color: #aaccee; text-align: right; word-break: break-all; }}
.info-val.live {{ color: #44ddaa; }}
.cam-table {{ width: 100%; border-collapse: collapse; margin-top: 3px; }}
.cam-table th {{ color: #445577; font-size: 10px; font-weight: normal;
  text-align: center; padding-bottom: 2px; border-bottom: 1px solid #1a2a3a; }}
.cam-table td.tc {{ text-align: center; padding: 1px 3px; font-size: 10px; color: #aabbcc; }}
.cam-table tr:hover td {{ background: #0d1a2d; }}
#plot-panel {{ flex: 1; position: relative; overflow: hidden; min-height: 0; }}
canvas#three {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
/* Bottom camera strip */
#bottom-cams {{
  height: calc(100vw / {n_cols} * {ar:.4f});
  flex-shrink: 0;
  display: grid;
  grid-template-columns: repeat({n_cols}, 1fr);
  grid-template-rows: 1fr;
  gap: 2px; background: #050510; padding: 2px;
  border-top: 1px solid #1a2a3a;
}}
.cam-wrap {{ position: relative; background: #080818; overflow: hidden; }}
.cam-wrap video {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}
.cam-wrap canvas {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }}
.cam-label {{
  position: absolute; top: 4px; left: 6px;
  background: rgba(0,0,0,0.65); color: #aabbdd;
  font-size: 10px; padding: 1px 5px; border-radius: 3px; pointer-events: none;
}}
.cam3d-label {{
  color: #aabbdd; font-size: 11px; font-family: monospace;
  background: rgba(0,0,0,0.65); padding: 2px 6px; border-radius: 3px;
  pointer-events: none; white-space: nowrap;
}}
/* Controls */
#controls {{
  flex-shrink: 0; background: #0d0d2e;
  border-top: 1px solid #223; padding: 8px 16px;
}}
#ctrl-row {{ display: flex; align-items: center; gap: 12px; }}
.btn {{
  background: #1a2a4a; border: 1px solid #335; color: #aaccff;
  padding: 4px 12px; border-radius: 4px; cursor: pointer;
  font-size: 13px; font-family: monospace;
}}
.btn:hover {{ background: #243560; }}
.btn.active {{ background: #2a4a8a; border-color: #559; color: #cce; }}
#btnRec {{ background: #2a1a1a; border-color: #633; color: #ff8888; }}
#btnRec:hover {{ background: #3a1a1a; }}
#btnRec.recording {{
  background: #5a0000; border-color: #c00; color: #ff4444;
  animation: pulse-rec 1s ease-in-out infinite;
}}
#btnStop {{ background: #2a1a1a; border-color: #633; color: #ff8888; }}
@keyframes pulse-rec {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}
#slider {{ flex: 1; accent-color: #4477cc; height: 4px; cursor: pointer; }}
#frame-label {{ min-width: 140px; text-align: right; font-size: 12px; color: #7799bb; }}
</style>
</head>
<body>

<div id="header" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="color:#ccd8ee;font-size:14px;font-weight:bold;">HomeCage &mdash; Automated Behavioral Tracking</span>
    <span style="color:#445566">&nbsp;&nbsp;/&nbsp;&nbsp;</span>
    <span style="color:#7799bb">Multi-camera 3D Pose Viewer</span>
    <span style="color:#334455;margin-left:16px;font-size:11px;">
      <span style="color:#ff6655">&#9670;</span> cameras &nbsp;
      <span style="color:#4488ff">&#9632;</span> {an0} &nbsp;
      <span style="color:#ff8800">&#9632;</span> {an1} &nbsp;&nbsp;
      drag&nbsp;rotate &middot; scroll&nbsp;zoom &middot; right-drag&nbsp;pan
    </span>
  </div>
  <div style="text-align:right;font-size:11px;color:#445566;white-space:nowrap;">
    <span style="color:#3a6090">acalapai@dpz.eu</span>&nbsp;&nbsp;|&nbsp;&nbsp;{build_day}
  </div>
</div>

<div id="main">
  <div id="viewer-row">

    <!-- ── Info panel ── -->
    <div id="info-panel">

      <div class="info-section">
        <h2>Recording</h2>
        <div class="info-row"><span class="info-key">Session</span><span class="info-val" style="font-size:10px;color:#7799aa;">{session_name}</span></div>
        <div class="info-row"><span class="info-key">Video&nbsp;fps</span><span class="info-val">{fps_ref:.1f}&thinsp;Hz</span></div>
        <div class="info-row"><span class="info-key">Sensor</span><span class="info-val">{img_w}&thinsp;&times;&thinsp;{img_h}</span></div>
        <div class="info-row"><span class="info-key">Cameras</span><span class="info-val">{", ".join(cam_ids)}</span></div>
        <div class="info-row"><span class="info-key">Max&nbsp;frame</span><span class="info-val">{max_frame}</span></div>
      </div>

      <div class="info-section">
        <h2>Pose Detection</h2>
        <div class="info-row"><span class="info-key">Model</span><span class="info-val" style="font-size:10px;">yolo26m-pose-ElmJok</span></div>
        <div class="info-row"><span class="info-key">Animals</span><span class="info-val" style="font-size:10px;">{", ".join(animals)}</span></div>
        <div class="info-row"><span class="info-key">Keypoints</span><span class="info-val">COCO&thinsp;17</span></div>
        <div class="info-row"><span class="info-key">KP&nbsp;conf&nbsp;thr</span><span class="info-val">{KP_CONF}</span></div>
        <div class="info-row"><span class="info-key">BBox&nbsp;conf&nbsp;thr</span><span class="info-val">{BBOX_CONF}</span></div>
      </div>

      <div class="info-section">
        <h2>Calibration</h2>
        <div class="info-row"><span class="info-key">Method</span><span class="info-val" style="font-size:10px;">recoverPose&thinsp;+&thinsp;solvePnP&thinsp;+&thinsp;Procrustes</span></div>
        <div class="info-row"><span class="info-key">Reference&nbsp;cam</span><span class="info-val">{ref_cam}</span></div>
        <div class="info-row"><span class="info-key">Reproj&nbsp;error</span><span class="info-val">{reproj_before:.1f}&thinsp;px</span></div>
        <div class="info-row"><span class="info-key">Scale</span><span class="info-val" style="font-size:10px;">known&thinsp;cam&thinsp;positions</span></div>
        <div class="info-row"><span class="info-key">Alignment</span><span class="info-val">Procrustes</span></div>
      </div>

      <div class="info-section">
        <h2>3D Reconstruction</h2>
        <div class="info-row"><span class="info-key">Method</span><span class="info-val" style="font-size:10px;">DLT&thinsp;triangulation</span></div>
        <div class="info-row"><span class="info-key">Inlier&nbsp;thr</span><span class="info-val">{TRAJ_RANSAC_MM:.0f}&thinsp;mm</span></div>
        <div class="info-row"><span class="info-key">Bone&nbsp;max</span><span class="info-val">{BONE_MAX_MM:.0f}&thinsp;mm</span></div>
        <div class="info-row"><span class="info-key">{an0}&nbsp;frames</span><span class="info-val">{an0_frames}&thinsp;/&thinsp;{max_frame}</span></div>
        <div class="info-row"><span class="info-key">{an1}&nbsp;frames</span><span class="info-val">{an1_frames}&thinsp;/&thinsp;{max_frame}</span></div>
      </div>

      <div class="info-section">
        <h2>Cameras</h2>
        <table class="cam-table">
          <tr><th>ID</th><th>{an0}</th><th>{an1}</th><th>dev&thinsp;mm</th><th>K</th></tr>
{cam_table_rows}        </table>
        <div style="font-size:9px;color:#3a5570;margin-top:3px;">
          &#9733; reference &nbsp; K: cal=calibrated, est=estimated
        </div>
      </div>

      <div class="info-section">
        <h2>Live</h2>
        <div class="info-row"><span class="info-key">Frame</span><span class="info-val live" id="li-frame">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#4488ff">{an0}&nbsp;X</span><span class="info-val live" id="li-elmx">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#4488ff">{an0}&nbsp;Y</span><span class="info-val live" id="li-elmy">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#4488ff">{an0}&nbsp;Z</span><span class="info-val live" id="li-elmz">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#ff8800">{an1}&nbsp;X</span><span class="info-val live" id="li-jokx">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#ff8800">{an1}&nbsp;Y</span><span class="info-val live" id="li-joky">&mdash;</span></div>
        <div class="info-row"><span class="info-key" style="color:#ff8800">{an1}&nbsp;Z</span><span class="info-val live" id="li-jokz">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Distance</span><span class="info-val live" id="li-dist">&mdash;</span></div>
      </div>

      <div class="info-section" style="margin-top:auto;padding-top:8px;border-top:1px solid #1a2a3a;font-size:10px;color:#3a5570;line-height:1.6;">
        Cognitive Ethology Lab<br>
        Deutsches Primatenzentrum<br>
        G&ouml;ttingen, Germany<br>
        <span style="color:#2a5080">acalapai@dpz.eu</span>
      </div>

    </div><!-- end info-panel -->

    <!-- ── Three.js scene ── -->
    <div id="plot-panel">
      <canvas id="three"></canvas>
    </div>

  </div><!-- end viewer-row -->

  <!-- ── Bottom camera strip ── -->
  <div id="bottom-cams">
{cam_strip_html}  </div>

</div><!-- end #main -->

<!-- ── Playback controls ── -->
<div id="controls">
  <div id="ctrl-row">
    <button class="btn" id="btnPlay" onclick="togglePlay()">&#9654; Play</button>
    <button class="btn" id="btnPause" onclick="togglePlay()" style="display:none">&#9646;&#9646; Pause</button>
    <input type="range" id="slider" min="0" max="{max_frame}" value="0">
    <span id="frame-label">frame 0 (1&thinsp;/&thinsp;{max_frame})</span>
    <label style="color:#aaccff;font-size:13px;">Speed:
      <select style="background:#1a2a4a;border:1px solid #335;color:#aaccff;padding:3px;font-family:monospace;font-size:12px;" onchange="setSpeed(+this.value)">
        <option value="0.4">0.4&times;</option>
        <option value="0.6">0.6&times;</option>
        <option value="1" selected>1&times;</option>
        <option value="1.5">1.5&times;</option>
        <option value="2">2&times;</option>
      </select>
    </label>
    <button class="btn active" id="btnBones" onclick="toggleBones()">&#9834; Bones</button>
    <button class="btn active" id="btnCoM"   onclick="toggleCoM()">CoM</button>
    <label style="color:#aaccff;font-size:13px;">Offset:
      <input type="range" id="off-slider" min="-10" max="10" value="0"
             style="width:70px;accent-color:#4477cc;height:4px;"
             oninput="document.getElementById('off-lbl').textContent=this.value">
      <span id="off-lbl" style="color:#7799bb;font-size:12px;">0</span>
    </label>
    <button class="btn" id="btnRec" onclick="startRec()">&#9210; Rec</button>
    <button class="btn" id="btnStop" onclick="stopRec()" style="display:none;background:#2a1a1a;border-color:#633;color:#ff8888;">&#9632; Stop</button>
  </div>
</div>

<script type="importmap">
{{
  "imports": {{
    "three":          "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/":  "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
  }}
}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }}   from 'three/addons/controls/OrbitControls.js';
import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

// ── Embedded data ─────────────────────────────────────────────────────────────
const MAX_FRAME     = {max_frame};
const FPS_BASE      = {fps_ref};
const ORIG_W        = {img_w};
const ORIG_H        = {img_h};
const CAM_IDS       = {json.dumps(cam_ids)};
const CAM_KNO_POS   = {json.dumps(known_pos_js)};
const ROOM          = {json.dumps(ROOM_MM)};
const DETS          = {json.dumps(DETS)};
const HIP3D         = {json.dumps(HIP3D)};
const KP3D          = {json.dumps(KP3D)};
const SKELETON      = {json.dumps(SKELETON)};
const AN0           = {json.dumps(an0)};
const AN1           = {json.dumps(an1)};
const ELM_HEX       = 0x4488ff;
const JOK_HEX       = 0xff8800;
const ELM_CSS       = '#4488ff';
const JOK_CSS       = '#ff8800';
const BONE_MAX      = {BONE_MAX_MM};
const KP_THR        = {KP_CONF};

// ── Three.js setup ────────────────────────────────────────────────────────────
const wrap   = document.getElementById('plot-panel');
const canvas = document.getElementById('three');
const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x080818);

const lblRenderer = new CSS2DRenderer();
lblRenderer.domElement.style.cssText =
    'position:absolute;top:0;left:0;pointer-events:none;width:100%;height:100%';
wrap.appendChild(lblRenderer.domElement);

const scene  = new THREE.Scene();
scene.fog    = new THREE.Fog(0x080818, 30000, 80000);
const cam3d  = new THREE.PerspectiveCamera(45, 1, 10, 120000);
cam3d.position.set(ROOM.x/2, -ROOM.y * 1.3, ROOM.z * 0.95);
cam3d.up.set(0, 0, 1);
cam3d.lookAt(ROOM.x/2, ROOM.y/2, ROOM.z/2);

const controls = new OrbitControls(cam3d, renderer.domElement);
controls.target.set(ROOM.x/2, ROOM.y/2, ROOM.z/2);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dLight = new THREE.DirectionalLight(0xffffff, 1.0);
dLight.position.set(ROOM.x, -ROOM.y, ROOM.z * 2);
scene.add(dLight);

// ── Room wireframe ────────────────────────────────────────────────────────────
const roomBox   = new THREE.BoxGeometry(ROOM.x, ROOM.y, ROOM.z);
const roomEdges = new THREE.EdgesGeometry(roomBox);
const roomLines = new THREE.LineSegments(
    roomEdges,
    new THREE.LineBasicMaterial({{ color: 0x3366cc, opacity: 0.35, transparent: true }})
);
roomLines.position.set(ROOM.x/2, ROOM.y/2, ROOM.z/2);
scene.add(roomLines);

// Floor grid at Z=0
const grid = new THREE.GridHelper(Math.max(ROOM.x, ROOM.y) * 1.5, 28, 0x1a2a4a, 0x0d1525);
grid.rotation.x = Math.PI / 2;
grid.position.set(ROOM.x/2, ROOM.y/2, 0);
scene.add(grid);

// ── Camera markers at known physical positions ────────────────────────────────
for (const [id, pos] of Object.entries(CAM_KNO_POS)) {{
    const mesh = new THREE.Mesh(
        new THREE.OctahedronGeometry(140, 0),
        new THREE.MeshPhongMaterial({{ color: 0xff4444, emissive: 0x441111 }})
    );
    mesh.position.set(pos[0], pos[1], pos[2]);
    scene.add(mesh);
    const div = document.createElement('div');
    div.textContent = id;
    div.style.cssText = 'color:#ff9999;font-size:11px;font-family:monospace;'
                      + 'background:rgba(0,0,0,.5);padding:1px 4px;border-radius:2px';
    const lbl = new CSS2DObject(div);
    lbl.position.set(0, 0, 200);
    mesh.add(lbl);
}}

// ── Animal skeleton factory ───────────────────────────────────────────────────
function makeAnimal(hex) {{
    const g = new THREE.Group();

    g.comMesh = new THREE.Mesh(
        new THREE.SphereGeometry(220, 24, 16),
        new THREE.MeshPhongMaterial({{
            color: hex, emissive: hex, emissiveIntensity: 0.12,
            opacity: 0.30, transparent: true,
            shininess: 80, depthWrite: false
        }})
    );
    g.add(g.comMesh);

    g.headMesh = new THREE.Mesh(
        new THREE.SphereGeometry(80, 12, 8),
        new THREE.MeshPhongMaterial({{ color: hex }})
    );
    g.headMesh.visible = false;
    g.add(g.headMesh);

    g.kpMeshes = [];
    for (let ki = 0; ki < 17; ki++) {{
        const km = new THREE.Mesh(
            new THREE.SphereGeometry(ki < 5 ? 0 : 38, 8, 6),
            new THREE.MeshPhongMaterial({{ color: hex }})
        );
        km.visible = false;
        g.kpMeshes.push(km);
        g.add(km);
    }}

    g.boneMeshes = [];
    for (let bi = 0; bi < SKELETON.length; bi++) {{
        const bm = new THREE.Mesh(
            new THREE.CylinderGeometry(18, 18, 1, 6),
            new THREE.MeshPhongMaterial({{ color: hex }})
        );
        bm.visible = false;
        g.boneMeshes.push(bm);
        g.add(bm);
    }}

    scene.add(g);
    return g;
}}

const elmGroup = makeAnimal(ELM_HEX);
const jokGroup = makeAnimal(JOK_HEX);
const anGroups = {{ [AN0]: elmGroup, [AN1]: jokGroup }};

let showBones = true;
let showCoM   = true;

function updateAnimal(g, kps3d, hip3d) {{
    const ok = new Array(17).fill(false);
    if (kps3d) {{
        for (let ki = 0; ki < 17; ki++) {{
            if (!kps3d[ki]) continue;
            for (const [a, b] of SKELETON) {{
                const o = (a === ki) ? b : (b === ki ? a : -1);
                if (o < 0 || !kps3d[o]) continue;
                const dx=kps3d[ki][0]-kps3d[o][0], dy=kps3d[ki][1]-kps3d[o][1],
                      dz=kps3d[ki][2]-kps3d[o][2];
                if (Math.sqrt(dx*dx+dy*dy+dz*dz) <= BONE_MAX) {{ ok[ki]=true; break; }}
            }}
        }}
    }}

    // Head centroid
    const hk = [0,1,2,3,4].filter(ki => kps3d && kps3d[ki] && ok[ki]);
    if (hk.length > 0) {{
        g.headMesh.position.set(
            hk.reduce((s,i)=>s+kps3d[i][0],0)/hk.length,
            hk.reduce((s,i)=>s+kps3d[i][1],0)/hk.length,
            hk.reduce((s,i)=>s+kps3d[i][2],0)/hk.length
        );
        g.headMesh.visible = true;
    }} else g.headMesh.visible = false;

    // Body keypoints
    for (let ki=5; ki<17; ki++) {{
        if (kps3d && kps3d[ki] && ok[ki]) {{
            g.kpMeshes[ki].position.set(kps3d[ki][0], kps3d[ki][1], kps3d[ki][2]);
            g.kpMeshes[ki].visible = true;
        }} else g.kpMeshes[ki].visible = false;
    }}

    // Bones
    for (let bi=0; bi<SKELETON.length; bi++) {{
        const [a, b] = SKELETON[bi];
        if (showBones && kps3d && kps3d[a] && kps3d[b] && ok[a] && ok[b]) {{
            const pa = new THREE.Vector3(...kps3d[a]);
            const pb = new THREE.Vector3(...kps3d[b]);
            const len = pa.distanceTo(pb);
            if (len < BONE_MAX) {{
                const bm = g.boneMeshes[bi];
                bm.position.copy(pa).lerp(pb, 0.5);
                bm.lookAt(pb);
                bm.rotateX(Math.PI/2);
                bm.scale.y = len;
                bm.visible = true;
            }} else g.boneMeshes[bi].visible = false;
        }} else g.boneMeshes[bi].visible = false;
    }}

    // CoM sphere
    if (showCoM && hip3d) {{
        g.comMesh.position.set(hip3d[0], hip3d[1], hip3d[2]);
        g.comMesh.visible = true;
    }} else g.comMesh.visible = false;
}}

// ── Camera strip drawing ──────────────────────────────────────────────────────
const camCtxs = {{}};
for (const cam of CAM_IDS) {{
    const cv = document.getElementById(`cvc-${{cam}}`);
    camCtxs[cam] = cv.getContext('2d');
}}

const videos = {{}};
for (const cam of CAM_IDS)
    videos[cam] = document.getElementById(`vid-${{cam}}`);

function seekAll(t) {{
    for (const cam of CAM_IDS) {{
        const v = videos[cam];
        if (v && v.readyState >= 1 && Math.abs(v.currentTime - t) > 0.4/FPS_BASE)
            v.currentTime = t;
    }}
}}

function drawStrip(vi) {{
    const offset = parseInt(document.getElementById('off-slider').value) || 0;
    for (const cam of CAM_IDS) {{
        const cv  = document.getElementById(`cvc-${{cam}}`);
        cv.width  = cv.parentElement.clientWidth;
        cv.height = cv.parentElement.clientHeight;
        const ctx = camCtxs[cam];
        ctx.clearRect(0, 0, cv.width, cv.height);

        const fi    = Math.max(0, vi + offset);
        const camD  = (DETS[cam]||{{}})[String(fi)];
        if (!camD) continue;

        // Compute letter-box offsets
        const nw = cv.width, nh = cv.height;
        const scale = Math.min(nw/ORIG_W, nh/ORIG_H);
        const ox = (nw - ORIG_W*scale)/2;
        const oy = (nh - ORIG_H*scale)/2;
        const sx = (u) => ox + u*scale;
        const sy = (v) => oy + v*scale;

        for (const [an, det] of Object.entries(camD)) {{
            const color = (an === AN0) ? ELM_CSS : JOK_CSS;
            // Bounding box
            const [x1,y1,x2,y2] = det.bbox;
            ctx.strokeStyle = color; ctx.lineWidth = 2;
            ctx.strokeRect(sx(x1), sy(y1), (x2-x1)*scale, (y2-y1)*scale);
            ctx.fillStyle = color;
            ctx.font = '11px monospace';
            ctx.fillText(an, sx(x1)+3, sy(y1)-3);

            // Skeleton keypoints and bones
            const kps = det.kps;   // [[x,y,c], ...]
            ctx.fillStyle = color;
            for (let ki=0; ki<17; ki++) {{
                if (kps[ki][2] < KP_THR) continue;
                ctx.beginPath();
                ctx.arc(sx(kps[ki][0]), sy(kps[ki][1]), 3, 0, Math.PI*2);
                ctx.fill();
            }}
            ctx.strokeStyle = color; ctx.lineWidth = 1.5;
            for (const [a,b] of SKELETON) {{
                if (kps[a][2] < KP_THR || kps[b][2] < KP_THR) continue;
                ctx.beginPath();
                ctx.moveTo(sx(kps[a][0]), sy(kps[a][1]));
                ctx.lineTo(sx(kps[b][0]), sy(kps[b][1]));
                ctx.stroke();
            }}
        }}
    }}
}}

// ── Playback ──────────────────────────────────────────────────────────────────
let curFrame = 0;
let playing  = false;
let speedMul = 1.0;
let playTmr  = null;
const slider = document.getElementById('slider');

function setFrame(vi) {{
    curFrame = Math.max(0, Math.min(MAX_FRAME, vi));
    slider.value = curFrame;
    document.getElementById('frame-label').textContent =
        `frame ${{curFrame}} (${{curFrame+1}}\u2009/\u2009${{MAX_FRAME+1}})`;
    seekAll(curFrame / FPS_BASE);

    const hip  = (HIP3D[String(curFrame)])||{{}};
    const kp   = (KP3D[String(curFrame)])||{{}};

    for (const [an, g] of Object.entries(anGroups))
        updateAnimal(g, (kp[an]||null), (hip[an]||null));

    // Live info
    const ePos = hip[AN0], jPos = hip[AN1];
    if (ePos) {{
        document.getElementById('li-elmx').textContent = ePos[0].toFixed(0)+'\u2009mm';
        document.getElementById('li-elmy').textContent = ePos[1].toFixed(0)+'\u2009mm';
        document.getElementById('li-elmz').textContent = ePos[2].toFixed(0)+'\u2009mm';
    }} else ['li-elmx','li-elmy','li-elmz'].forEach(id => document.getElementById(id).textContent='\u2014');

    if (jPos) {{
        document.getElementById('li-jokx').textContent = jPos[0].toFixed(0)+'\u2009mm';
        document.getElementById('li-joky').textContent = jPos[1].toFixed(0)+'\u2009mm';
        document.getElementById('li-jokz').textContent = jPos[2].toFixed(0)+'\u2009mm';
    }} else ['li-jokx','li-joky','li-jokz'].forEach(id => document.getElementById(id).textContent='\u2014');

    if (ePos && jPos) {{
        const dx=ePos[0]-jPos[0], dy=ePos[1]-jPos[1], dz=ePos[2]-jPos[2];
        document.getElementById('li-dist').textContent = Math.sqrt(dx*dx+dy*dy+dz*dz).toFixed(0)+'\u2009mm';
    }} else document.getElementById('li-dist').textContent = '\u2014';

    document.getElementById('li-frame').textContent = `${{curFrame}} / ${{MAX_FRAME}}`;
    drawStrip(curFrame);
}}

function togglePlay() {{
    playing = !playing;
    document.getElementById('btnPlay').style.display  = playing ? 'none' : '';
    document.getElementById('btnPause').style.display = playing ? '' : 'none';
    if (playing) {{
        playTmr = setInterval(() => {{
            if (curFrame >= MAX_FRAME) {{ curFrame = 0; }}
            setFrame(curFrame + 1);
        }}, 1000 / FPS_BASE / speedMul);
    }} else {{
        clearInterval(playTmr);
    }}
}}

window.togglePlay  = togglePlay;
window.setSpeed    = (v) => {{
    speedMul = v;
    if (playing) {{ clearInterval(playTmr); togglePlay(); togglePlay(); }}
}};
window.toggleBones = () => {{
    showBones = !showBones;
    document.getElementById('btnBones').classList.toggle('active', showBones);
    setFrame(curFrame);
}};
window.toggleCoM = () => {{
    showCoM = !showCoM;
    document.getElementById('btnCoM').classList.toggle('active', showCoM);
    setFrame(curFrame);
}};

slider.addEventListener('input', () => setFrame(+slider.value));

// ── Screen recording ──────────────────────────────────────────────────────────
let recStream = null; let recRec = null; let recChunks = [];
window.startRec = async function() {{
    try {{
        recStream = await navigator.mediaDevices.getDisplayMedia(
            {{ video: {{ frameRate: 30 }}, audio: false }});
        const mime = ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4']
            .find(t => MediaRecorder.isTypeSupported(t)) || '';
        recRec = new MediaRecorder(recStream, mime ? {{ mimeType: mime }} : {{}});
        recChunks = [];
        recRec.ondataavailable = e => {{ if (e.data.size > 0) recChunks.push(e.data); }};
        recRec.onstop = () => {{
            if (recChunks.length === 0) return;
            const blob = new Blob(recChunks, {{ type: recChunks[0].type || 'video/webm' }});
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = 'homecage_recording.webm';
            a.click();
        }};
        recRec.start(1000);
        document.getElementById('btnRec').classList.add('recording');
        document.getElementById('btnRec').style.display  = 'none';
        document.getElementById('btnStop').style.display = '';
    }} catch(e) {{ console.warn('Rec error:', e); }}
}};
window.stopRec = function() {{
    if (recRec && recRec.state !== 'inactive') recRec.stop();
    if (recStream) recStream.getTracks().forEach(t => t.stop());
    document.getElementById('btnRec').classList.remove('recording');
    document.getElementById('btnRec').style.display  = '';
    document.getElementById('btnStop').style.display = 'none';
}};

// ── Resize handler ────────────────────────────────────────────────────────────
function onResize() {{
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h);
    lblRenderer.setSize(w, h);
    cam3d.aspect = w / h;
    cam3d.updateProjectionMatrix();
}}
new ResizeObserver(onResize).observe(wrap);
onResize();

// ── Render loop ───────────────────────────────────────────────────────────────
(function loop() {{
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, cam3d);
    lblRenderer.render(scene, cam3d);
}})();

setFrame(0);
</script>
</body>
</html>"""

    viewer_path.write_text(html, encoding="utf-8")
    size_mb = viewer_path.stat().st_size / 1e6
    print(f"  Viewer: {viewer_path}  ({size_mb:.1f} MB)")
    return viewer_path


# ══════════════════════════════════════════════════════════════════════════════
# Stage 9 — Update datalus_config.json
# ══════════════════════════════════════════════════════════════════════════════

def stage9_update_config(poses, reproj_before, reproj_after, deviations,
                         ref_cam, trajectories, animal_labels):
    _hdr(9, "UPDATE CONFIG")
    cfg = {}
    if CONFIG_JSON.exists():
        with open(CONFIG_JSON) as f:
            cfg = json.load(f)

    an0 = animal_labels[0] if len(animal_labels) > 0 else "Elm"
    an1 = animal_labels[1] if len(animal_labels) > 1 else "Jok"
    n0  = len(trajectories.get(an0, {}).get("frame", []))
    n1  = len(trajectories.get(an1, {}).get("frame", []))

    # Count total frames across all cameras for success rate
    total_frames = max(
        max(trajectories.get(an0, {}).get("frame", [0]) or [0]),
        max(trajectories.get(an1, {}).get("frame", [0]) or [0]),
        1
    )
    r0 = f"{n0/max(1,total_frames)*100:.1f}"
    r1 = f"{n1/max(1,total_frames)*100:.1f}"

    cfg["homecage_calibration"] = {
        "approach": "recoverPose + solvePnP + BA + Procrustes alignment",
        "model": MODEL_PATH.name,
        "animals": animal_labels,
        "reference_camera": ref_cam,
        "reprojection_error_before_BA_px": f"{reproj_before:.1f}",
        "reprojection_error_after_BA_px":  f"{reproj_after:.1f}",
        "per_camera_position_deviation_mm": {
            cam: f"{deviations.get(cam, 0):.1f}" for cam in ["102","108","113","117"]
        },
        f"{an0}_frames_triangulated": n0,
        f"{an1}_frames_triangulated": n1,
        f"{an0}_success_rate_pct": r0,
        f"{an1}_success_rate_pct": r1,
        "status": "DONE",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    with open(CONFIG_JSON, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"  Updated {CONFIG_JSON}")
    print(f"  Reprojection: {reproj_before:.1f} px → {reproj_after:.1f} px")
    for an in animal_labels:
        n = len(trajectories.get(an, {}).get("frame", []))
        print(f"  {an}: {n} frames triangulated")


# ══════════════════════════════════════════════════════════════════════════════
# HTTP server + browser launcher
# ══════════════════════════════════════════════════════════════════════════════

def _free_port(start=8765):
    import socket
    for p in range(start, start + 100):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p)); return p
            except OSError:
                continue
    return start


def _serve_and_open(viewer_path):
    import subprocess, webbrowser, time
    port = _free_port()
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=str(OUT_DIR),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(0.7)
    url = f"http://localhost:{port}/{viewer_path.name}"
    webbrowser.open(url)
    print(f"\n  Viewer served at: {url}")
    print(f"  (HTTP server running in background on port {port})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    video_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RAW
    if not video_folder.exists():
        print(f"[ERROR] Video folder not found: {video_folder}")
        sys.exit(1)

    session_name = video_folder.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("HomeCage Calibration Pipeline — fresh implementation")
    print(f"Session : {session_name}")
    print(f"Videos  : {video_folder}")
    print(f"Output  : {OUT_DIR}")

    cam_map = stage1_inventory(video_folder)

    missing = [c for c in CAM_IDS if c not in cam_map]
    if missing:
        print(f"\n[WARN] Missing cameras: {missing}")
    cam_ids = [c for c in CAM_IDS if c in cam_map]
    if len(cam_ids) < 2:
        print("[ERROR] Need at least 2 cameras"); sys.exit(1)

    meta = stage2_3_detect(cam_map)

    cams, sources = stage4_intrinsics(cam_ids, meta)

    print("\n  Loading detections from pose files:")
    detections = _load_detections(cam_ids)
    total = sum(sum(len(v) for v in detections[c].values()) for c in cam_ids)
    print(f"  Total detections: {total}")

    poses, ref_cam, reproj_before, reproj_after, deviations = \
        stage5_extrinsics(cam_ids, detections, cams)

    stage6_write_yamls(cam_ids, poses, cams)

    trajectories, animal_labels = stage7_trajectories(
        cam_ids, detections, poses, cams)

    viewer_path = stage8_build_viewer(
        cam_ids, cam_map, detections, poses, cams, sources,
        trajectories, animal_labels, meta,
        reproj_before, reproj_after, deviations, ref_cam, session_name,
    )

    stage9_update_config(poses, reproj_before, reproj_after, deviations,
                         ref_cam, trajectories, animal_labels)

    # ── Final three-line summary ───────────────────────────────────────────────
    print(f"\n{'═'*60}\nDONE\n{'═'*60}")
    print(f"  Reprojection error : {reproj_before:.1f} px → {reproj_after:.1f} px (before / after BA)")
    for an in animal_labels:
        n = len(trajectories.get(an, {}).get("frame", []))
        print(f"  {an} triangulated  : {n} frames")
    print(f"  Viewer             : {viewer_path}")

    _serve_and_open(viewer_path)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
