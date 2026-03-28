#!/usr/bin/env python3
"""
PriCaB_HumanCalib.py  —  PriCaB Human Calibration Pipeline

Single-script: raw videos → ABT YAMLs + interactive HTML viewer

Usage:
    python3 PriCaB_HumanCalib.py <video_folder>

Stages:
  1  Inspect     — scan videos, detect camera IDs, report metadata
  2  Extract     — 1 frame per 500 ms  →  PriCaB_output/frames/{cam_id}/
  3  Pose        — YOLOv8-pose on every frame  →  PriCaB_output/pose_{cam_id}.txt
  4  Intrinsics  — load from intrinsics.npz or estimate from lens specs
  5  Extrinsics  — essmat + recoverPose + chaining  →  pricab_poses.npz
  6  YAMLs       — ABT-compatible  →  PriCaB_output/yamls/{cam_id}.yaml
  7  Viewer      — self-contained HTML  →  PriCaB_output/pricab_viewer.html
  8  Config      — append results to datalus_config.json
"""

import base64
import json
import re
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
INTRINSICS_NPZ  = BASE_DIR / "DatalusCalibration" / "intrinsics.npz"
CONFIG_JSON     = BASE_DIR / "datalus_config.json"
POSE_MODEL_PATH = BASE_DIR / "yolov8n-pose.pt"
OUT_DIR         = BASE_DIR / "PriCaB_output"

# ─── Constants ─────────────────────────────────────────────────────────────────
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}
FRAME_INTERVAL_MS = 500      # 1 frame per 500 ms
BBOX_CONF_THRESH  = 0.3      # minimum bbox confidence for a detection
KP_CONF_THRESH    = 0.5      # minimum keypoint confidence for correspondence
RANSAC_THRESH     = 0.001    # essmat RANSAC threshold (normalised coords)
MIN_DIRECT_CORR   = 500      # min keypoint pairs for direct (non-chained) solve
LENS_FOCAL_MM     = 8.0      # fallback focal length (C-mount)
SENSOR_WIDTH_MM   = 11.2     # fallback sensor width (Sony IMX304)
IMG_SCALE         = 0.25     # viewer image resize factor
JPEG_QUALITY      = 40       # viewer JPEG compression quality
PERSON_HEIGHT_MM  = 1700.0   # known person height for metric scale
SCALE_KPS         = [0, 5, 6, 11, 12, 15, 16]  # nose,lsh,rsh,lhip,rhip,lank,rank
SCALE_KP_CONF     = 0.5      # keypoint confidence for scale frames
SCALE_BBOX_CONF   = 0.7      # bbox confidence for scale frames
SCALE_VERT_FRAC   = 0.30     # min vertical span / img height for full-body frames


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _header(n, title):
    print(f"\n{'═'*60}")
    print(f"STAGE {n} — {title}")
    print('═'*60)


def _detect_cam_ids(video_files):
    """
    Find the numeric token that is unique per filename (and differs across
    files) — that token is the camera ID.
    Returns {cam_id_str: Path}.
    """
    if not video_files:
        return {}
    stems    = [Path(f).stem for f in video_files]
    num_runs = [re.findall(r'\d+', s) for s in stems]
    n        = len(video_files)
    max_pos  = max((len(r) for r in num_runs), default=0)

    id_pos = None
    for pos in range(max_pos):
        vals = [r[pos] if pos < len(r) else None for r in num_runs]
        if None not in vals and len(set(vals)) == n:
            id_pos = pos
            break

    result = {}
    for vf, runs in zip(video_files, num_runs):
        cid = runs[id_pos] if (id_pos is not None and id_pos < len(runs)) else Path(vf).stem
        result[cid] = Path(vf)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Inspect
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_inspect(video_folder):
    _header(1, "INSPECT")
    folder = Path(video_folder)
    if not folder.exists():
        sys.exit(f"[ERROR] Folder not found: {folder}")

    videos     = sorted(p for p in folder.iterdir() if p.suffix in VIDEO_EXTS)
    cam_videos = _detect_cam_ids(videos)
    print(f"  Found {len(cam_videos)} camera(s): {sorted(cam_videos)}")

    if len(cam_videos) < 2:
        sys.exit("[ERROR] At least 2 cameras are required for calibration.")

    cam_info = {}
    for cam_id, vpath in sorted(cam_videos.items()):
        cap = cv2.VideoCapture(str(vpath))
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        dur        = nf / fps
        n_extract  = int(dur * 1000 / FRAME_INTERVAL_MS) + 1
        print(f"  cam {cam_id}: {w}×{h}  {fps:.1f}fps  {nf}fr  {dur:.1f}s"
              f"  → ~{n_extract} frames to extract  [{vpath.name}]")
        cam_info[cam_id] = {"path": vpath, "w": w, "h": h,
                             "fps": fps, "n_frames": nf, "duration_s": dur}

    return cam_videos, cam_info


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Extract frames
# ═══════════════════════════════════════════════════════════════════════════════

def stage2_extract_frames(cam_videos, cam_info):
    _header(2, "EXTRACT FRAMES  (1 frame per 500 ms)")
    OUT_DIR.mkdir(exist_ok=True)
    frames_dirs = {}

    for cam_id, vpath in sorted(cam_videos.items()):
        out = OUT_DIR / "frames" / cam_id
        frames_dirs[cam_id] = out

        existing = sorted(out.glob(f"{cam_id}_frame_*.png"))
        if existing:
            print(f"  cam {cam_id}: {len(existing)} frames already exist — skip")
            continue

        out.mkdir(parents=True, exist_ok=True)
        fps  = cam_info[cam_id]["fps"]
        step = max(1, int(round(fps * FRAME_INTERVAL_MS / 1000.0)))
        cap  = cv2.VideoCapture(str(vpath))
        idx  = saved = fnum = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if fnum % step == 0:
                cv2.imwrite(str(out / f"{cam_id}_frame_{idx:06d}.png"), frame)
                idx  += 1
                saved += 1
            fnum += 1

        cap.release()
        print(f"  cam {cam_id}: extracted {saved} frames → {out}")

    return frames_dirs


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Human pose detection
# ═══════════════════════════════════════════════════════════════════════════════

def _pose_line(frame_idx, x1, y1, x2, y2, bbox_conf, kps):
    """One detection per line: frame x1 y1 x2 y2 bbox_conf kp0x kp0y kp0c …"""
    kp_str = " ".join(
        f"{kps[i, 0]:.2f} {kps[i, 1]:.2f} {kps[i, 2]:.4f}"
        for i in range(17)
    )
    return (f"{frame_idx} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}"
            f" {bbox_conf:.4f} {kp_str}")


def stage3_detect_pose(cam_ids, frames_dirs):
    _header(3, "HUMAN POSE DETECTION  (YOLOv8-pose)")
    model = None

    for cam_id in sorted(cam_ids):
        out_txt = OUT_DIR / f"pose_{cam_id}.txt"
        if out_txt.exists():
            n = sum(1 for ln in out_txt.read_text().splitlines() if ln.strip())
            print(f"  cam {cam_id}: {out_txt.name} already exists ({n} lines) — skip")
            continue

        if model is None:
            print("  Loading YOLOv8-pose model (downloads yolov8n-pose.pt if absent)…")
            from ultralytics import YOLO  # type: ignore
            src = str(POSE_MODEL_PATH) if POSE_MODEL_PATH.exists() else "yolov8n-pose.pt"
            model = YOLO(src)

        frame_files = sorted(
            frames_dirs[cam_id].glob(f"{cam_id}_frame_*.png"))
        if not frame_files:
            print(f"  cam {cam_id}: no frames found — skip")
            continue

        lines = []
        for img_path in frame_files:
            m = re.search(rf"{re.escape(cam_id)}_frame_(\d+)\.png", img_path.name)
            if not m:
                continue
            fidx    = int(m.group(1))
            results = model(str(img_path), verbose=False)
            res     = results[0]

            if res.boxes is None or len(res.boxes) == 0:
                continue
            confs = res.boxes.conf.cpu().numpy()
            best  = int(np.argmax(confs))
            if confs[best] < BBOX_CONF_THRESH:
                continue

            xyxy = res.boxes.xyxy.cpu().numpy()[best]
            if res.keypoints is not None and len(res.keypoints.data) > 0:
                kps = res.keypoints.data.cpu().numpy()[best]   # (17, 3)
            else:
                kps = np.zeros((17, 3), dtype=np.float32)

            lines.append(_pose_line(fidx, *xyxy, confs[best], kps))

        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"  cam {cam_id}: {len(lines)} detections → {out_txt.name}")


def _load_pose_files(cam_ids):
    """Returns {cam_id: {frame_idx: {'bbox', 'bbox_conf', 'kps'}}}."""
    detections = {}
    for cam_id in sorted(cam_ids):
        cam_dets = {}
        txt = OUT_DIR / f"pose_{cam_id}.txt"
        if txt.exists():
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 57:          # 1 + 5 + 51
                    continue
                fidx      = int(cols[0])
                bbox      = [float(c) for c in cols[1:5]]
                bbox_conf = float(cols[5])
                kps       = np.array(cols[6:57], dtype=np.float32).reshape(17, 3)
                cam_dets[fidx] = {"bbox": bbox, "bbox_conf": bbox_conf, "kps": kps}
        detections[cam_id] = cam_dets
        print(f"  cam {cam_id}: {len(cam_dets)} detections loaded")
    return detections


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Intrinsics
# ═══════════════════════════════════════════════════════════════════════════════

def stage4_intrinsics(cam_ids, cam_info):
    _header(4, "INTRINSICS")

    loaded_K    = {}
    loaded_dist = {}
    if INTRINSICS_NPZ.exists():
        npz         = np.load(str(INTRINSICS_NPZ))
        loaded_K    = {k[:-2]: npz[k] for k in npz.files if k.endswith("_K")}
        loaded_dist = {k[:-5]: npz[k] for k in npz.files if k.endswith("_dist")}

    cams    = {}
    sources = {}
    for cam_id in sorted(cam_ids):
        w, h = cam_info[cam_id]["w"], cam_info[cam_id]["h"]

        if cam_id in loaded_K:
            K    = loaded_K[cam_id].astype(np.float64)
            dist = loaded_dist.get(cam_id, np.zeros((1, 5))).ravel().astype(np.float64)
            src  = "loaded from DatalusCalibration/intrinsics.npz"
        else:
            fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) * w
            K  = np.array([[fx, 0, w / 2.0],
                           [0, fx, h / 2.0],
                           [0,  0,      1.0]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            src  = (f"estimated (f={LENS_FOCAL_MM}mm lens, "
                    f"{SENSOR_WIDTH_MM}mm sensor → fx={fx:.0f}px)")

        cams[cam_id]    = {"K": K, "dist": dist, "w": w, "h": h}
        sources[cam_id] = src
        print(f"  cam {cam_id}: fx={K[0,0]:.1f}px  cx={K[0,2]:.1f}"
              f"  cy={K[1,2]:.1f}  — {src}")

    return cams, sources


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Extrinsics
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_corr(ca, cb, detections):
    """Return pixel correspondences for a camera pair (same frame index)."""
    shared = sorted(set(detections[ca]) & set(detections[cb]))
    pts_a, pts_b = [], []
    for f in shared:
        kps_a = detections[ca][f]["kps"]
        kps_b = detections[cb][f]["kps"]
        for ki in range(17):
            if kps_a[ki, 2] >= KP_CONF_THRESH and kps_b[ki, 2] >= KP_CONF_THRESH:
                pts_a.append([kps_a[ki, 0], kps_a[ki, 1]])
                pts_b.append([kps_b[ki, 0], kps_b[ki, 1]])
    return (np.array(pts_a, dtype=np.float64),
            np.array(pts_b, dtype=np.float64),
            len(shared))


def _solve_rel_pose(pts_a, pts_b, K_a, dist_a, K_b, dist_b):
    """
    Relative pose: p_b = R @ p_a + T  (normalised coords, RANSAC essmat).
    Returns (R, T_1d, inlier_rate) or None if not enough data / failed.
    """
    if len(pts_a) < 8:
        return None

    n_a = cv2.undistortPoints(
        pts_a.reshape(-1, 1, 2), K_a, dist_a).reshape(-1, 2)
    n_b = cv2.undistortPoints(
        pts_b.reshape(-1, 1, 2), K_b, dist_b).reshape(-1, 2)

    try:
        E, mask = cv2.findEssentialMat(
            n_a, n_b,
            np.eye(3, dtype=np.float64),  # already normalised
            method=cv2.RANSAC,
            threshold=RANSAC_THRESH,
            prob=0.999,
        )
    except cv2.error:
        return None

    if E is None or mask is None:
        return None

    inlier_rate = float(mask.ravel().sum()) / len(pts_a)
    _, R, T, _  = cv2.recoverPose(
        E, n_a, n_b, np.eye(3, dtype=np.float64), mask=mask)
    return R, T.ravel(), inlier_rate


def _get_rel(pairwise, ca, cb):
    """
    Retrieve relative pose from pairwise dict regardless of stored direction.
    Returns (R_ab, T_ab_1d) such that p_b = R_ab @ p_a + T_ab, or None.
    """
    if (ca, cb) in pairwise:
        R, T, *_ = pairwise[(ca, cb)]
        return R, T
    if (cb, ca) in pairwise:
        R_ba, T_ba, *_ = pairwise[(cb, ca)]
        R_ab = R_ba.T
        T_ab = -R_ba.T @ T_ba
        return R_ab, T_ab
    return None


def _corr_count(pairwise, ca, cb):
    if (ca, cb) in pairwise:
        return pairwise[(ca, cb)][2]
    if (cb, ca) in pairwise:
        return pairwise[(cb, ca)][2]
    return 0


def stage5_extrinsics(cam_ids, detections, cams):
    _header(5, "EXTRINSICS  (essmat + recoverPose)")

    poses_npz = OUT_DIR / "pricab_poses.npz"
    cam_ids_s = sorted(cam_ids)

    if poses_npz.exists():
        print(f"  {poses_npz.name} already exists — loading, skipping computation")
        npz = np.load(str(poses_npz))
        poses   = {}
        ref_cam = str(npz["reference_camera"][0])
        for cid in cam_ids_s:
            R = npz[f"{cid}_R"]
            T = npz[f"{cid}_T"].reshape(3, 1)
            K = cams[cid]["K"]
            C = (-R.T @ T).ravel()
            P = K @ np.hstack([R, T])
            poses[cid] = {"R": R, "T": T, "P": P, "C": C}
        reproj_err = float(npz["reprojection_error_px"][0])
        print(f"  reference={ref_cam}, reproj={reproj_err:.1f}px")
        return poses, ref_cam, reproj_err

    # ── Detection counts → reference camera ───────────────────────────────────
    det_counts = {c: len(detections[c]) for c in cam_ids_s}
    ref_cam    = max(det_counts, key=det_counts.get)
    print(f"  Detection counts: {det_counts}")
    print(f"  Reference camera: {ref_cam}")

    # ── Pairwise correspondences & essmat ─────────────────────────────────────
    pairwise = {}
    print("\n  Pairwise analysis:")
    for ca, cb in combinations(cam_ids_s, 2):
        pts_a, pts_b, n_shared = _collect_corr(ca, cb, detections)
        n_corr = len(pts_a)
        if n_corr < 8:
            print(f"    {ca}↔{cb}: {n_shared} shared frames, {n_corr} kp pairs — insufficient")
            continue
        result = _solve_rel_pose(
            pts_a, pts_b,
            cams[ca]["K"], cams[ca]["dist"],
            cams[cb]["K"], cams[cb]["dist"],
        )
        if result is None:
            print(f"    {ca}↔{cb}: essmat solve failed")
            continue
        R_ab, T_ab, inlier_rate = result
        n_inliers = int(round(inlier_rate * n_corr))
        pairwise[(ca, cb)] = (R_ab, T_ab, n_corr, n_inliers, inlier_rate)
        print(f"    {ca}↔{cb}: {n_shared} frames, {n_corr} kp pairs,"
              f" {n_inliers} inliers ({inlier_rate*100:.1f}%)")

    # ── BFS to build absolute poses (world = reference camera frame) ──────────
    poses     = {ref_cam: {"R": np.eye(3), "T": np.zeros((3, 1))}}
    remaining = [c for c in cam_ids_s if c != ref_cam]
    max_iter  = len(remaining) * (len(cam_ids_s) + 1)

    for _ in range(max_iter):
        if not remaining:
            break
        for cam in list(remaining):
            # Direct from reference
            rel = _get_rel(pairwise, ref_cam, cam)
            direct_corr = _corr_count(pairwise, ref_cam, cam)

            if rel is not None and direct_corr >= MIN_DIRECT_CORR:
                R_rc, T_rc = rel
                poses[cam] = {"R": R_rc, "T": T_rc.reshape(3, 1)}
                remaining.remove(cam)
                print(f"  cam {cam}: direct from {ref_cam}"
                      f" ({direct_corr} kp pairs)")
                break   # restart inner loop after modifying remaining

            # Chain through the best-connected solved intermediate
            best_mid, best_score = None, -1
            for mid in poses:
                if mid == cam:
                    continue
                if _get_rel(pairwise, mid, cam) is None:
                    continue
                score = _corr_count(pairwise, mid, cam)
                if score > best_score:
                    best_score, best_mid = score, mid

            if best_mid is not None:
                R_m   = poses[best_mid]["R"]
                T_m   = poses[best_mid]["T"].ravel()
                R_mc, T_mc = _get_rel(pairwise, best_mid, cam)
                R_abs = R_mc @ R_m
                T_abs = R_mc @ T_m + T_mc
                poses[cam] = {"R": R_abs, "T": T_abs.reshape(3, 1)}
                remaining.remove(cam)
                print(f"  cam {cam}: chained via {best_mid}"
                      f" ({best_score} kp pairs to intermediate)")
                break

    for cam in remaining:
        print(f"  [WARN] cam {cam}: could not solve — using identity pose")
        poses[cam] = {"R": np.eye(3), "T": np.zeros((3, 1))}

    # ── Build P matrices and camera centres ───────────────────────────────────
    for cid in cam_ids_s:
        R = poses[cid]["R"]
        T = poses[cid]["T"]
        K = cams[cid]["K"]
        poses[cid]["P"] = K @ np.hstack([R, T])
        poses[cid]["C"] = (-R.T @ T).ravel()

    # ── Reprojection error ────────────────────────────────────────────────────
    reproj_err = _reprojection_error(cam_ids_s, detections, poses, cams)
    print(f"\n  Reprojection error (hip midpoint, median): {reproj_err:.1f} px")

    # ── Save ──────────────────────────────────────────────────────────────────
    save = {
        "reference_camera":       np.array([ref_cam]),
        "reprojection_error_px":  np.array([reproj_err]),
    }
    for cid in cam_ids_s:
        save[f"{cid}_R"] = poses[cid]["R"]
        save[f"{cid}_T"] = poses[cid]["T"]
    np.savez(str(poses_npz), **save)
    print(f"  Saved → {poses_npz}")

    return poses, ref_cam, reproj_err


def _triangulate_hip(kps_per_cam, poses):
    """
    Triangulate left_hip (kp11) + right_hip (kp12), return midpoint or None.
    Uses pairwise cv2.triangulatePoints with cheirality filter.
    """
    results = []
    for hip_idx in [11, 12]:
        obs = {}
        for cam_id, kps in kps_per_cam.items():
            x, y, c = kps[hip_idx]
            if c >= KP_CONF_THRESH:
                obs[cam_id] = (float(x), float(y))
        if len(obs) < 2:
            continue

        pts3d = []
        for ca, cb in combinations(obs, 2):
            xa, ya = obs[ca]
            xb, yb = obs[cb]
            X4 = cv2.triangulatePoints(
                poses[ca]["P"], poses[cb]["P"],
                np.array([[xa], [ya]], dtype=np.float64),
                np.array([[xb], [yb]], dtype=np.float64),
            )
            if abs(float(X4[3])) < 1e-10:
                continue
            X3 = (X4[:3] / X4[3]).ravel()
            if not np.all(np.isfinite(X3)):
                continue
            Ra, Ta = poses[ca]["R"], poses[ca]["T"].ravel()
            Rb, Tb = poses[cb]["R"], poses[cb]["T"].ravel()
            if (Ra @ X3 + Ta)[2] > 0 and (Rb @ X3 + Tb)[2] > 0:
                pts3d.append(X3)

        if pts3d:
            results.append(np.median(pts3d, axis=0))

    if len(results) < 2:
        return None
    return (results[0] + results[1]) / 2.0


def _reprojection_error(cam_ids, detections, poses, cams):
    """Triangulate hip midpoint per frame, project back, return median px error."""
    errors = []
    all_frames = set()
    for c in cam_ids:
        all_frames |= set(detections[c])

    for frame in sorted(all_frames):
        kps_per_cam = {c: detections[c][frame]["kps"]
                       for c in cam_ids if frame in detections[c]}
        if len(kps_per_cam) < 2:
            continue
        X3 = _triangulate_hip(kps_per_cam, poses)
        if X3 is None:
            continue

        for cam_id, kps in kps_per_cam.items():
            for hip_idx in [11, 12]:
                x, y, c = kps[hip_idx]
                if c < KP_CONF_THRESH:
                    continue
                R, T, K = poses[cam_id]["R"], poses[cam_id]["T"].ravel(), cams[cam_id]["K"]
                dist    = cams[cam_id]["dist"]
                rvec, _ = cv2.Rodrigues(R)
                proj, _ = cv2.projectPoints(
                    X3.reshape(1, 1, 3), rvec, T, K, dist)
                px, py  = proj.ravel()
                errors.append(float(np.hypot(px - x, py - y)))

    return float(np.median(errors)) if errors else 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Write YAMLs
# ═══════════════════════════════════════════════════════════════════════════════

def stage6_write_yamls(cam_ids, cams, poses):
    _header(6, "WRITE YAMLs  (ABT-compatible)")
    yaml_dir = OUT_DIR / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in sorted(cam_ids):
        out = yaml_dir / f"{cam_id}.yaml"
        if out.exists():
            print(f"  cam {cam_id}: {out.name} already exists — skip")
            continue

        K    = cams[cam_id]["K"]
        dist = cams[cam_id]["dist"]
        R    = poses[cam_id]["R"]
        T    = poses[cam_id]["T"]

        # ABT reads K.mat().T and R.mat().T → store K^T and R^T
        fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
        fs.write("intrinsicMatrix",           K.T)
        fs.write("distortionCoefficients",    dist.reshape(1, -1))
        fs.write("R",                         R.T)
        fs.write("T",                         T.reshape(3, 1))
        fs.release()
        print(f"  cam {cam_id}: → {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Build viewer
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_image(path, scale, quality):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w  = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def stage7_build_viewer(cam_ids, cam_info, detections, cams, poses,
                         frames_dirs, ref_cam):
    _header(7, "BUILD VIEWER  (self-contained HTML)")
    out_html = OUT_DIR / "pricab_viewer.html"
    if out_html.exists():
        print(f"  {out_html.name} already exists — skip")
        return out_html

    cam_ids_s = sorted(cam_ids)
    native_w  = cam_info[cam_ids_s[0]]["w"]
    native_h  = cam_info[cam_ids_s[0]]["h"]
    disp_w    = int(native_w * IMG_SCALE)
    disp_h    = int(native_h * IMG_SCALE)

    # ── Frames with ≥1 detection ──────────────────────────────────────────────
    all_frames = set()
    for c in cam_ids_s:
        all_frames |= set(detections[c])
    shared_frames = sorted(all_frames)

    print(f"  {len(shared_frames)} frames with detections, "
          f"{len(cam_ids_s)} cameras")

    # ── Triangulate hip midpoints ─────────────────────────────────────────────
    positions3d = {}
    for frame in shared_frames:
        kps_per_cam = {c: detections[c][frame]["kps"]
                       for c in cam_ids_s if frame in detections[c]}
        if len(kps_per_cam) < 2:
            continue
        X3 = _triangulate_hip(kps_per_cam, poses)
        if X3 is not None and np.all(np.isfinite(X3)) and np.linalg.norm(X3) < 50.0:
            positions3d[frame] = [X3[0], X3[2], -X3[1]]

    print(f"  Hip midpoint triangulated: {len(positions3d)}/{len(shared_frames)} frames")

    # ── Encode images ─────────────────────────────────────────────────────────
    print(f"  Encoding {len(shared_frames) * len(cam_ids_s)} images "
          f"(scale={IMG_SCALE}, q={JPEG_QUALITY})…")
    images = {c: [] for c in cam_ids_s}
    for frame in shared_frames:
        for c in cam_ids_s:
            img_path = frames_dirs[c] / f"{c}_frame_{frame:06d}.png"
            images[c].append(_encode_image(img_path, IMG_SCALE, JPEG_QUALITY))
    print("  Encoding done.")

    # ── Bounding boxes for JS ─────────────────────────────────────────────────
    det_js = {}
    for frame in shared_frames:
        fd = {}
        for c in cam_ids_s:
            if frame in detections[c]:
                fd[c] = detections[c][frame]["bbox"]
        det_js[frame] = fd

    # ── Coordinate rotation: OpenCV cam space (Y-down, Z-forward) → Z-up world
    # Rx(-90°): (x, y, z) → (x, z, -y)  — floor becomes XY plane, Z points up
    def _to_zup(v):
        return [v[0], v[2], -v[1]]

    # ── Camera centres for 3D ─────────────────────────────────────────────────
    cam_centres = {c: _to_zup(poses[c]["C"].tolist()) for c in cam_ids_s}

    # ── Plotly traces ─────────────────────────────────────────────────────────
    first_pos = next(
        (positions3d[f] for f in shared_frames if f in positions3d),
        cam_centres[ref_cam])

    plotly_traces = [
        {
            "type": "scatter3d",
            "x": [cam_centres[c][0] for c in cam_ids_s],
            "y": [cam_centres[c][1] for c in cam_ids_s],
            "z": [cam_centres[c][2] for c in cam_ids_s],
            "mode": "markers+text",
            "name": "Cameras",
            "text": [f"cam{c}" for c in cam_ids_s],
            "textposition": "top center",
            "textfont": {"color": "white", "size": 11},
            "marker": {"color": "red", "size": 8, "symbol": "diamond",
                       "line": {"color": "white", "width": 1}},
        },
        {
            "type": "scatter3d",
            "x": [first_pos[0]], "y": [first_pos[1]], "z": [first_pos[2]],
            "mode": "markers+text",
            "name": "Hip midpoint",
            "text": ["person"],
            "textposition": "top center",
            "textfont": {"color": "#44aaff", "size": 13},
            "marker": {"color": "#44aaff", "size": 14, "symbol": "circle",
                       "line": {"color": "white", "width": 2}},
        },
    ]

    # ── Grid layout ───────────────────────────────────────────────────────────
    n_cams    = len(cam_ids_s)
    grid_cols = min(n_cams, 2)
    grid_rows = (n_cams + grid_cols - 1) // grid_cols

    cam_divs = "\n    ".join(
        f'<div class="cam-wrap"><canvas id="c{c}"></canvas>'
        f'<div class="cam-label">cam {c}</div></div>'
        for c in cam_ids_s
    )
    canvas_js_obj = "{\n" + ",\n".join(
        f"  '{c}': document.getElementById('c{c}')" for c in cam_ids_s
    ) + "\n}"

    # ── JSON blobs ────────────────────────────────────────────────────────────
    j_frames     = json.dumps(shared_frames)
    j_images     = json.dumps(images)
    j_detections = json.dumps(det_js)
    j_positions  = json.dumps({str(k): v for k, v in positions3d.items()})
    j_traces     = json.dumps(plotly_traces)
    j_cam_ids    = json.dumps(cam_ids_s)
    j_img_dims   = json.dumps({"native_w": native_w, "native_h": native_h,
                                "disp_w": disp_w, "disp_h": disp_h})
    j_layout = json.dumps({
        "paper_bgcolor": "#0a0a1a",
        "font": {"color": "white", "family": "monospace"},
        "scene": {
            "bgcolor": "#0d0d22",
            "xaxis": {"title": "X (floor)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "yaxis": {"title": "Y (floor)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "zaxis": {"title": "Z (up)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2},
                       "up": {"x": 0, "y": 0, "z": 1}},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
        "legend": {"bgcolor": "rgba(20,20,50,0.85)", "font": {"color": "white"}},
        "uirevision": "keep",
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PriCaB — Human Calibration Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  display: flex; flex-direction: column; height: 100vh;
  background: #0a0a1a; color: #ccc; font-family: monospace; overflow: hidden;
}}
#header {{
  padding: 8px 16px; background: #0d0d2e; border-bottom: 1px solid #223;
  font-size: 14px; color: #8899bb; flex-shrink: 0;
}}
#main {{ display: flex; flex: 1; overflow: hidden; }}
#cam-panel {{
  width: 50%; display: grid;
  grid-template-columns: repeat({grid_cols}, 1fr);
  grid-template-rows: repeat({grid_rows}, 1fr);
  gap: 2px; background: #050510; padding: 2px;
}}
.cam-wrap {{ position: relative; background: #080818; overflow: hidden; }}
.cam-wrap canvas {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.cam-label {{
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.65); color: #aabbdd;
  font-size: 11px; padding: 2px 6px; border-radius: 3px; pointer-events: none;
}}
#plot-panel {{ flex: 1; position: relative; }}
#plot3d {{ width: 100%; height: 100%; }}
#controls {{
  flex-shrink: 0; background: #0d0d2e;
  border-top: 1px solid #223; padding: 10px 16px;
}}
#ctrl-row {{ display: flex; align-items: center; gap: 12px; }}
.btn {{
  background: #1a2a4a; border: 1px solid #335; color: #aaccff;
  padding: 5px 14px; border-radius: 4px; cursor: pointer;
  font-size: 13px; font-family: monospace;
}}
.btn:hover {{ background: #243560; }}
#slider {{ flex: 1; accent-color: #4477cc; height: 4px; cursor: pointer; }}
#frame-label {{ min-width: 140px; text-align: right; font-size: 12px; color: #7799bb; }}
</style>
</head>
<body>
<div id="header">
  PriCaB — Human Calibration Viewer &nbsp;|&nbsp;
  <span style="color:#ff4444">&#9670; Cameras</span> &nbsp;
  <span style="color:#44aaff">&#11044; Hip midpoint</span>
</div>
<div id="main">
  <div id="cam-panel">
    {cam_divs}
  </div>
  <div id="plot-panel"><div id="plot3d"></div></div>
</div>
<div id="controls">
  <div id="ctrl-row">
    <button class="btn" id="btnPlay">&#9654; Play</button>
    <button class="btn" id="btnPause">&#9646;&#9646; Pause</button>
    <input type="range" id="slider" min="0" max="{len(shared_frames)-1}" value="0">
    <span id="frame-label">frame {shared_frames[0] if shared_frames else 0}</span>
  </div>
</div>

<script>
const FRAMES     = {j_frames};
const IMAGES     = {j_images};
const DETECTIONS = {j_detections};
const POSITIONS  = {j_positions};
const IMG_DIMS   = {j_img_dims};
const CAM_IDS    = {j_cam_ids};

Plotly.newPlot('plot3d', {j_traces}, {j_layout}, {{
  scrollZoom: true, displayModeBar: true,
  modeBarButtonsToRemove: ['toImage'], displaylogo: false,
}});

const canvases = {canvas_js_obj};
const imgCache = {{}};

function getImg(camId, frameIdx) {{
  const key = camId + '_' + frameIdx;
  if (!imgCache[key]) {{
    const im = new Image();
    im.src = IMAGES[camId][frameIdx];
    imgCache[key] = im;
  }}
  return imgCache[key];
}}

(function preload() {{
  for (let i = 0; i < Math.min(4, FRAMES.length); i++)
    CAM_IDS.forEach(c => getImg(c, i));
}})();

function drawCanvas(camId, frameIdx) {{
  const canvas = canvases[camId];
  if (!canvas) return;
  const ctx  = canvas.getContext('2d');
  const dw   = canvas.offsetWidth  || IMG_DIMS.disp_w;
  const dh   = canvas.offsetHeight || IMG_DIMS.disp_h;
  canvas.width  = dw;
  canvas.height = dh;
  const img    = getImg(camId, frameIdx);
  const scaleX = dw / IMG_DIMS.native_w;
  const scaleY = dh / IMG_DIMS.native_h;
  function draw() {{
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(img, 0, 0, dw, dh);
    const frame = FRAMES[frameIdx];
    const bbox  = (DETECTIONS[frame] || {{}})[camId];
    if (bbox) {{
      const x1 = bbox[0]*scaleX, y1 = bbox[1]*scaleY;
      const x2 = bbox[2]*scaleX, y2 = bbox[3]*scaleY;
      ctx.strokeStyle = '#44aaff';
      ctx.lineWidth   = Math.max(2, dw * 0.003);
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      ctx.fillStyle = '#44aaff';
      ctx.font      = Math.round(dw * 0.035) + 'px monospace';
      ctx.fillText('person', x1+3, Math.max(y1-4, 14));
    }}
  }}
  if (img.complete) draw(); else img.onload = draw;
}}

function update3D(frameIdx) {{
  const frame = FRAMES[frameIdx].toString();
  const pos   = POSITIONS[frame];
  Plotly.restyle('plot3d', {{
    x: [pos ? [pos[0]] : [null]],
    y: [pos ? [pos[1]] : [null]],
    z: [pos ? [pos[2]] : [null]],
  }}, [1]);
}}

function setFrame(idx) {{
  CAM_IDS.forEach(c => drawCanvas(c, idx));
  update3D(idx);
  document.getElementById('frame-label').textContent =
    'frame ' + FRAMES[idx] + ' (' + (idx+1) + '/' + FRAMES.length + ')';
  document.getElementById('slider').value = idx;
  for (let i = idx+1; i < Math.min(idx+4, FRAMES.length); i++)
    CAM_IDS.forEach(c => getImg(c, i));
}}

const slider = document.getElementById('slider');
slider.addEventListener('input', () => setFrame(+slider.value));

let playTimer = null, playIdx = 0;
document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playTimer) return;
  if (playIdx >= FRAMES.length - 1) playIdx = 0;
  playTimer = setInterval(() => {{
    playIdx++;
    setFrame(playIdx);
    if (playIdx >= FRAMES.length - 1) {{
      clearInterval(playTimer); playTimer = null;
    }}
  }}, 80);
}});
document.getElementById('btnPause').addEventListener('click', () => {{
  clearInterval(playTimer); playTimer = null;
  playIdx = +slider.value;
}});

setFrame(0);
window.addEventListener('resize', () => setFrame(+slider.value));
</script>
</body>
</html>"""

    out_html.write_text(html, encoding="utf-8")
    size_mb = out_html.stat().st_size / 1e6
    print(f"  Written → {out_html}  ({size_mb:.1f} MB)")
    return out_html


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Update config
# ═══════════════════════════════════════════════════════════════════════════════

def stage8_update_config(video_folder, cam_ids, cam_info, ref_cam,
                          reproj_err, intrinsics_sources):
    _header(8, "UPDATE CONFIG  (datalus_config.json)")
    if not CONFIG_JSON.exists():
        print(f"  {CONFIG_JSON} not found — skip")
        return

    cfg   = json.loads(CONFIG_JSON.read_text())
    cam_s = sorted(cam_ids)
    w     = cam_info[cam_s[0]]["w"]
    h     = cam_info[cam_s[0]]["h"]

    cfg["pricab_human_test"] = {
        "data_folder":            str(video_folder),
        "cameras_found":          cam_s,
        "resolution":             f"{w}x{h}",
        "approach":               "human pose keypoints, YOLOv8-pose",
        "intrinsics_source":      {c: intrinsics_sources[c] for c in cam_s},
        "reference_camera":       ref_cam,
        "reprojection_error_px":  f"{reproj_err:.1f}",
        "status":                 "DONE",
        "timestamp":              datetime.now().strftime("%Y-%m-%d %H:%M"),
    }

    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"  Updated → {CONFIG_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    DEFAULT_FOLDER = BASE_DIR / "Measurements" / "250404_HumanTest_2"
    video_folder   = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_FOLDER)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       PriCaB — Human Calibration Pipeline               ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Input : {video_folder}")
    print(f"  Output: {OUT_DIR}")

    cam_videos, cam_info   = stage1_inspect(video_folder)
    cam_ids                = sorted(cam_videos)

    frames_dirs            = stage2_extract_frames(cam_videos, cam_info)

    stage3_detect_pose(cam_ids, frames_dirs)
    print(f"\n  Loading pose detections…")
    detections             = _load_pose_files(cam_ids)

    cams, intrinsics_sources = stage4_intrinsics(cam_ids, cam_info)

    poses, ref_cam, reproj_err = stage5_extrinsics(cam_ids, detections, cams)

    stage6_write_yamls(cam_ids, cams, poses)

    viewer_path = stage7_build_viewer(
        cam_ids, cam_info, detections, cams, poses, frames_dirs, ref_cam)

    stage8_update_config(
        video_folder, cam_ids, cam_info, ref_cam,
        reproj_err, intrinsics_sources)

    print(f"\n{'═'*60}")
    print(f"DONE — {len(cam_ids)} cameras calibrated"
          f" | reproj: {reproj_err:.1f}px"
          f" | viewer: {viewer_path}")
    print('═'*60)

    import os
    os.system(f"open '{viewer_path}'")


if __name__ == "__main__":
    main()
