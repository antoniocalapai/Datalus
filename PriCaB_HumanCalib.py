#!/usr/bin/env python3
"""
PriCaB_HumanCalib.py  —  PriCaB Human Calibration Pipeline  (v1.0 stable)
===========================================================================

Marker-free, human-body-based multi-camera calibration.
A single script turns raw synchronised videos into metric camera poses
(ABT-compatible YAML files) and an interactive 3-D HTML viewer.

No checkerboard, no fiducials — just a person walking through the scene.

----------------------------------------------------------------------------
QUICK START
----------------------------------------------------------------------------
    python3 PriCaB_HumanCalib.py  <video_folder>

    <video_folder>  must contain one video file per camera, all recorded
    simultaneously.  Camera IDs are auto-detected from file names.

    Example:
        python3 PriCaB_HumanCalib.py Measurements/250404_HumanTest_2

    All outputs land in  PriCaB_output/  next to this script.
    The viewer opens automatically in the default browser when done.

----------------------------------------------------------------------------
PIPELINE STAGES
----------------------------------------------------------------------------
  Stage 1  INSPECT
           Scan the folder, auto-detect camera IDs from file names, read
           video metadata (resolution, fps, frame count, duration).

  Stage 2  EXTRACT FRAMES                          (skipped if done)
           Sample 1 frame per 500 ms from every video.
           Output: PriCaB_output/frames/<cam_id>/<cam_id>_frame_NNNNNN.png

  Stage 3  HUMAN POSE DETECTION                    (skipped if done)
           Run YOLOv8-pose on every extracted frame (yolov8n-pose.pt,
           downloaded automatically from ultralytics if absent).
           Output: PriCaB_output/pose_<cam_id>.txt
           Format per line:
             frame_idx  x1 y1 x2 y2  bbox_conf
             kp0_x kp0_y kp0_conf … kp16_x kp16_y kp16_conf
           COCO 17-keypoint layout (0=nose … 16=right_ankle).

  Stage 4  INTRINSICS
           Load per-camera K and distortion from
           DatalusCalibration/intrinsics.npz  (produced by DatulusCalib
           steps 1–2).  Falls back to a thin-lens estimate from the
           constants LENS_FOCAL_MM and SENSOR_WIDTH_MM if a camera is
           absent from the .npz.

  Stage 5  EXTRINSICS — metric reconstruction
           Eight-step pipeline:

           5.1  Anchor pair — find the camera pair with the most
                co-detected full-body frames (bbox_conf ≥ 0.70,
                all SCALE_KPS visible with conf ≥ 0.50,
                vertical span ≥ 30 % of image height).

           5.2  recoverPose — essential matrix + chirality on the anchor
                pair; this yields a unit-scale relative pose.

           5.3  Metric scale — person height constraint:
                scale = PERSON_HEIGHT_MM /
                        median(head-to-ankle distance in raw units).
                Applied immediately to the anchor T vector so the entire
                reconstruction is in millimetres from this point on.

           5.4  Triangulate landmarks — SCALE_KPS triangulated from the
                anchor pair into metric 3-D; cheirality-filtered.

           5.5  solvePnPRansac — every remaining camera is placed by
                matching its 2-D keypoint observations to the metric 3-D
                landmarks; cameras are processed in overlap-descending
                order so each solved camera can contribute new landmarks
                for the next (iterative expansion).

           5.6  Bundle adjustment — scipy least_squares (TRF solver,
                soft-L1 loss) jointly refines all non-anchor camera
                poses with the 3-D landmarks fixed.

           5.7  Room alignment — rigid transform (translate + Ry yaw)
                that places the centroid of cameras 106 and 110 at
                viewer (X=0, Y=0) and rotates so cameras 105/107 lie
                along the positive viewer-X axis.

           5.8  Save — pricab_poses.npz + pricab_poses_scaled.npz
                (both contain the same metric poses after alignment).

  Stage 6  WRITE YAMLs
           One ABT-compatible YAML per camera under
           PriCaB_output/yamls/<cam_id>.yaml.
           Fields: camera_matrix, dist_coeffs, rotation_matrix,
           translation_vector, image_size.

  Stage 7  BUILD VIEWER
           Self-contained HTML (no server required):
           - Left pane: Plotly 3-D scatter of camera positions + hip
             midpoint trajectory (one dot per frame, animatable).
           - Right pane: 2×N grid of per-camera live video with YOLO
             bounding box overlay for the current frame.
           - Failed cameras (could not be placed) are excluded
             automatically from both panes.
           Output: PriCaB_output/pricab_viewer.html

  Stage 8  UPDATE CONFIG
           Writes / updates datalus_config.json with camera positions
           and calibration metadata.

----------------------------------------------------------------------------
COORDINATE CONVENTIONS
----------------------------------------------------------------------------
  OpenCV / raw camera space:   X right, Y down, Z forward (into scene)
  Viewer / world space:        X right, Y into-scene (floor), Z up
  Conversion (Rx −90°):        viewer = (raw_x, raw_z, −raw_y)

  Room alignment is expressed as a Ry rotation in raw space, which
  corresponds to a Rz rotation in viewer space:
    same wall  →  same viewer-Y coordinate
    positive viewer-X  →  direction from cams 106/110 toward cams 105/107

  All distances are in millimetres after stage 5.3.

----------------------------------------------------------------------------
OUTPUTS
----------------------------------------------------------------------------
  PriCaB_output/
    frames/<cam_id>/          — extracted PNG frames
    pose_<cam_id>.txt         — YOLO keypoint detections
    pricab_poses.npz          — metric camera poses (R, T per camera)
    pricab_poses_scaled.npz   — identical copy (kept for backward compat)
    yamls/<cam_id>.yaml       — ABT-compatible calibration files
    pricab_viewer.html        — interactive 3-D viewer (self-contained)
  datalus_config.json         — updated with camera pose metadata

----------------------------------------------------------------------------
SKIP / RESUME LOGIC
----------------------------------------------------------------------------
  Stages 2 and 3 are skipped if their outputs already exist on disk —
  re-running the script after a crash or partial run will resume from
  stage 4 automatically.  To force a full re-run, delete the relevant
  files in PriCaB_output/.

  Stage 7 is skipped if pricab_viewer.html already exists.  Delete it
  to force a viewer rebuild without re-running detection.

----------------------------------------------------------------------------
DEPENDENCIES
----------------------------------------------------------------------------
  opencv-python   ≥ 4.8
  numpy           ≥ 1.24
  scipy           ≥ 1.11
  ultralytics     ≥ 8.0    (YOLOv8; optional — only needed for stage 3)

----------------------------------------------------------------------------
TUNABLE CONSTANTS  (top of file)
----------------------------------------------------------------------------
  PERSON_HEIGHT_MM   — known subject height in mm  (default: 1700)
  SCALE_KPS          — keypoint indices used for scale & landmarks
                       [nose, l/r-shoulder, l/r-hip, l/r-ankle]
  SCALE_BBOX_CONF    — min detection confidence for full-body frames
  SCALE_VERT_FRAC    — min vertical body span / image height
  FRAME_INTERVAL_MS  — frame sampling interval (ms)
  LENS_FOCAL_MM      — fallback focal length (mm)
  SENSOR_WIDTH_MM    — fallback sensor width (mm)
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
    """
    Scan *video_folder* for camera videos and return metadata.

    Camera IDs are detected automatically: the numeric token that is
    unique and differs across all file names in the folder is taken as
    the camera ID (e.g. "109" from "cam_109_20250404.mp4").

    Returns
    -------
    cam_videos : dict[str, Path]
        Mapping camera-ID → video file path.
    cam_info : dict[str, dict]
        Per-camera metadata: path, w, h, fps, n_frames, duration_s.
    """
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
    """
    Extract one PNG frame per FRAME_INTERVAL_MS milliseconds from each video.

    Frames are written to PriCaB_output/frames/<cam_id>/ and named
    <cam_id>_frame_NNNNNN.png with a zero-based sequential index.

    Skipped entirely for cameras whose output directory already contains
    matching PNG files (safe to re-run after interruption).

    Returns
    -------
    frames_dirs : dict[str, Path]
        Mapping camera-ID → directory containing the extracted frames.
    """
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
    """
    Run YOLOv8-pose on every extracted frame and write detection files.

    Model: yolov8n-pose.pt (downloaded automatically on first run via
    ultralytics if not present at POSE_MODEL_PATH).

    For each frame, keeps only the highest-confidence bounding box.
    Detections below BBOX_CONF_THRESH are discarded.

    Output format (PriCaB_output/pose_<cam_id>.txt):
        One detection per line:
          frame_idx  x1 y1 x2 y2  bbox_conf
          kp0_x kp0_y kp0_conf … kp16_x kp16_y kp16_conf
        Keypoints follow the COCO 17-point layout:
          0 nose  1 l-eye  2 r-eye  3 l-ear  4 r-ear
          5 l-shoulder  6 r-shoulder  7 l-elbow  8 r-elbow
          9 l-wrist  10 r-wrist  11 l-hip  12 r-hip
          13 l-knee  14 r-knee  15 l-ankle  16 r-ankle

    Skipped for any camera whose pose_<cam_id>.txt already exists.
    """
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
    """
    Load or estimate per-camera intrinsic parameters.

    Priority order:
      1. DatalusCalibration/intrinsics.npz  (produced by DatulusCalib
         steps 1–2 using a physical checkerboard).  Keys: <cam_id>_K
         (3×3 float64) and <cam_id>_dist (1×5 float64).
      2. Thin-lens estimate:
           fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) × image_width_px
           K  = diag(fx, fx, 1) with principal point at image centre
           dist = zeros(5)

    Returns
    -------
    cams : dict[str, dict]
        Per-camera: K (3×3), dist (5,), w (px), h (px).
    sources : dict[str, str]
        Human-readable string describing where each camera's intrinsics
        came from (for the stage 8 config log).
    """
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
    """
    Metric reconstruction:
      1. Anchor pair   — best pair by valid full-body co-detections → recoverPose
      2. Scale         — person height constraint applied immediately to anchor T
      3. Landmarks     — triangulate SCALE_KPS from anchor pair in metric mm
      4. solvePnP      — place every remaining camera against metric landmarks,
                         chain outward and accumulate new landmarks each pass
      5. BA            — scipy least_squares jointly refines all non-anchor poses
      6. Save          — pricab_poses.npz and pricab_poses_scaled.npz (metric)
    """
    _header(5, "EXTRINSICS — anchor pair + solvePnP + bundle adjustment")

    poses_npz  = OUT_DIR / "pricab_poses.npz"
    scaled_npz = OUT_DIR / "pricab_poses_scaled.npz"
    cam_ids_s  = sorted(cam_ids)
    IMG_H      = cams[cam_ids_s[0]]["h"]

    # ── helpers ───────────────────────────────────────────────────────────────
    def _valid_det(det, kc=SCALE_KP_CONF):
        if det["bbox_conf"] < SCALE_BBOX_CONF:
            return False
        kps = det["kps"]
        for ki in SCALE_KPS:
            if kps[ki, 2] < kc:
                return False
        ys = [kps[ki, 1] for ki in SCALE_KPS]
        return (max(ys) - min(ys)) >= SCALE_VERT_FRAC * IMG_H

    def _tri(Pa, Pb, xa, ya, xb, yb):
        X4 = cv2.triangulatePoints(
            Pa, Pb,
            np.array([[xa], [ya]], dtype=np.float64),
            np.array([[xb], [yb]], dtype=np.float64))
        w = float(X4[3])
        if abs(w) < 1e-10:
            return None
        X3 = (X4[:3] / w).ravel()
        return X3 if np.all(np.isfinite(X3)) else None

    def _make_P(cam_id, pose):
        return cams[cam_id]["K"] @ np.hstack([pose["R"], pose["T"]])

    def _project_np(X3_arr, rvec, tvec, K, dist):
        """Numpy batch projection — faster than per-point cv2.projectPoints."""
        R, _ = cv2.Rodrigues(rvec)
        pts   = (R @ X3_arr.T).T + tvec.ravel()   # (N, 3) camera-space
        z     = pts[:, 2:3]
        xy    = pts[:, :2] / np.where(np.abs(z) < 1e-10, 1e-10, z)
        r2    = xy[:, 0]**2 + xy[:, 1]**2
        k1, k2 = dist[0], dist[1]
        p1, p2 = dist[2], dist[3]
        k3     = dist[4] if len(dist) > 4 else 0.0
        rad    = 1 + k1*r2 + k2*r2**2 + k3*r2**3
        dx     = 2*p1*xy[:, 0]*xy[:, 1] + p2*(r2 + 2*xy[:, 0]**2)
        dy     = p1*(r2 + 2*xy[:, 1]**2) + 2*p2*xy[:, 0]*xy[:, 1]
        xd, yd = xy[:, 0]*rad + dx, xy[:, 1]*rad + dy
        u = K[0, 0]*xd + K[0, 2]
        v = K[1, 1]*yd + K[1, 2]
        return np.stack([u, v], axis=1)  # (N, 2)

    # ── 1. Find best anchor pair ──────────────────────────────────────────────
    pair_valid = {}
    kp_conf_used = SCALE_KP_CONF
    for kc in [SCALE_KP_CONF, 0.3]:
        for ca, cb in combinations(cam_ids_s, 2):
            shared = sorted(set(detections[ca]) & set(detections[cb]))
            vf = [f for f in shared
                  if f in detections[ca] and f in detections[cb]
                  and _valid_det(detections[ca][f], kc)
                  and _valid_det(detections[cb][f], kc)]
            if vf:
                pair_valid[(ca, cb)] = vf
        if pair_valid:
            kp_conf_used = kc
            break

    if not pair_valid:
        print("  [WARN] No valid full-body pairs — falling back to all-keypoint essmat")
        # Degenerate: run old approach with best-overlap pair
        best = max(combinations(cam_ids_s, 2),
                   key=lambda p: len(set(detections[p[0]]) & set(detections[p[1]])))
        pair_valid[best] = sorted(set(detections[best[0]]) & set(detections[best[1]]))

    anchor_pair = max(pair_valid, key=lambda p: len(pair_valid[p]))
    ca, cb = anchor_pair
    vf_anchor = pair_valid[anchor_pair]
    print(f"\n  Anchor pair : {ca}↔{cb}  ({len(vf_anchor)} valid full-body frames,"
          f"  kp_conf={kp_conf_used})")

    # Per-pair coverage summary (top 8)
    top = sorted(pair_valid.items(), key=lambda x: -len(x[1]))[:8]
    for (a, b), frames in top:
        mark = " ← anchor" if (a, b) == anchor_pair else ""
        print(f"    {a}↔{b}: {len(frames)} frames{mark}")

    # ── 2. recoverPose on anchor pair ─────────────────────────────────────────
    pts_a, pts_b = [], []
    for f in vf_anchor:
        kps_a = detections[ca][f]["kps"]
        kps_b = detections[cb][f]["kps"]
        for ki in range(17):
            if kps_a[ki, 2] >= kp_conf_used and kps_b[ki, 2] >= kp_conf_used:
                pts_a.append([kps_a[ki, 0], kps_a[ki, 1]])
                pts_b.append([kps_b[ki, 0], kps_b[ki, 1]])

    pts_a = np.array(pts_a, dtype=np.float64)
    pts_b = np.array(pts_b, dtype=np.float64)
    Ka, da = cams[ca]["K"], cams[ca]["dist"]
    Kb, db = cams[cb]["K"], cams[cb]["dist"]

    n_a = cv2.undistortPoints(pts_a.reshape(-1, 1, 2), Ka, da).reshape(-1, 2)
    n_b = cv2.undistortPoints(pts_b.reshape(-1, 1, 2), Kb, db).reshape(-1, 2)

    E, mask_e = cv2.findEssentialMat(n_a, n_b, np.eye(3),
                                      method=cv2.RANSAC,
                                      threshold=RANSAC_THRESH, prob=0.999)
    inliers_e = int(mask_e.ravel().sum())
    print(f"\n  recoverPose: {len(pts_a)} corr, {inliers_e} inliers "
          f"({100*inliers_e/len(pts_a):.0f}%)")

    _, R_cb, T_cb_unit, _ = cv2.recoverPose(E, n_a, n_b, np.eye(3), mask=mask_e)
    T_cb_unit = T_cb_unit.ravel()

    poses = {
        ca: {"R": np.eye(3, dtype=np.float64), "T": np.zeros((3, 1), dtype=np.float64)},
        cb: {"R": R_cb.astype(np.float64),     "T": T_cb_unit.reshape(3, 1)},
    }

    # ── 3. Person-height scale on anchor pair ─────────────────────────────────
    Pa_u = _make_P(ca, poses[ca])
    Pb_u = _make_P(cb, poses[cb])
    Ra, Ta = poses[ca]["R"], poses[ca]["T"].ravel()
    Rb, Tb = poses[cb]["R"], poses[cb]["T"].ravel()

    ha_dists = []
    for f in vf_anchor:
        kps_a = detections[ca][f]["kps"]
        kps_b = detections[cb][f]["kps"]
        pts3d = {}
        ok = True
        for ki in SCALE_KPS:
            X3 = _tri(Pa_u, Pb_u,
                      float(kps_a[ki, 0]), float(kps_a[ki, 1]),
                      float(kps_b[ki, 0]), float(kps_b[ki, 1]))
            if X3 is None or (Ra @ X3 + Ta)[2] <= 0 or (Rb @ X3 + Tb)[2] <= 0:
                ok = False; break
            pts3d[ki] = X3
        if not ok:
            continue
        ha = float(np.linalg.norm(pts3d[0] - (pts3d[15] + pts3d[16]) / 2))
        sw = float(np.linalg.norm(pts3d[5] - pts3d[6]))
        if ha > 0:
            ha_dists.append(ha)

    if not ha_dists:
        print("  [WARN] No valid head-ankle measurements — scale defaults to 1")
        scale = 1.0
    else:
        scale = PERSON_HEIGHT_MM / float(np.median(ha_dists))
        sw_scaled = float(np.median([
            np.linalg.norm(
                _tri(Pa_u, Pb_u,
                     float(detections[ca][f]["kps"][5, 0]),
                     float(detections[ca][f]["kps"][5, 1]),
                     float(detections[cb][f]["kps"][5, 0]),
                     float(detections[cb][f]["kps"][5, 1])) -
                _tri(Pa_u, Pb_u,
                     float(detections[ca][f]["kps"][6, 0]),
                     float(detections[ca][f]["kps"][6, 1]),
                     float(detections[cb][f]["kps"][6, 0]),
                     float(detections[cb][f]["kps"][6, 1]))
            )
            for f in vf_anchor[:10]
            if _tri(Pa_u, Pb_u,
                    float(detections[ca][f]["kps"][5, 0]),
                    float(detections[ca][f]["kps"][5, 1]),
                    float(detections[cb][f]["kps"][5, 0]),
                    float(detections[cb][f]["kps"][5, 1])) is not None
        ] or [0])) * scale

    print(f"  Scale factor: {scale:.4f}  "
          f"(median head-ankle = {float(np.median(ha_dists)) if ha_dists else 0:.4f} units)")
    if ha_dists:
        print(f"  Shoulder width validation: {sw_scaled:.0f} mm  "
              f"({'✓' if 350 <= sw_scaled <= 500 else '⚠ expected 350–500 mm'})")

    # Apply scale to anchor cb T
    poses[cb]["T"] = poses[cb]["T"] * scale

    # ── 4. Triangulate metric landmarks from anchor pair ──────────────────────
    Pa = _make_P(ca, poses[ca])
    Pb = _make_P(cb, poses[cb])
    Ra, Ta = poses[ca]["R"], poses[ca]["T"].ravel()
    Rb, Tb = poses[cb]["R"], poses[cb]["T"].ravel()

    landmarks = {}   # (frame, ki) → X3_mm
    for f in vf_anchor:
        kps_a = detections[ca][f]["kps"]
        kps_b = detections[cb][f]["kps"]
        for ki in SCALE_KPS:
            X3 = _tri(Pa, Pb,
                      float(kps_a[ki, 0]), float(kps_a[ki, 1]),
                      float(kps_b[ki, 0]), float(kps_b[ki, 1]))
            if X3 is None:
                continue
            if (Ra @ X3 + Ta)[2] > 0 and (Rb @ X3 + Tb)[2] > 0:
                landmarks[(f, ki)] = X3

    print(f"  Anchor landmarks: {len(landmarks)}  "
          f"({len(vf_anchor)} frames × up to {len(SCALE_KPS)} kps)")

    # ── 5. solvePnP for remaining cameras ─────────────────────────────────────
    # Sort candidates by overlap with current landmark frames (best first)
    placed    = {ca, cb}
    remaining = [c for c in cam_ids_s if c not in placed]
    lm_frames = set(f for f, _ in landmarks)

    def _lm_overlap(c):
        return sum(1 for f in lm_frames
                   if f in detections[c]
                   and any(detections[c][f]["kps"][ki, 2] >= kp_conf_used
                           for ki in SCALE_KPS))

    # Iterative placement: each pass may unlock new cameras via new landmarks
    max_passes = len(remaining) + 1
    for _pass in range(max_passes):
        if not remaining:
            break
        remaining.sort(key=_lm_overlap, reverse=True)
        placed_this_pass = []

        for cam in list(remaining):
            # Build 2D–3D correspondences from existing landmarks
            obj_pts, img_pts = [], []
            for (f, ki), X3 in landmarks.items():
                if f not in detections[cam]:
                    continue
                conf = detections[cam][f]["kps"][ki, 2]
                if conf < kp_conf_used:
                    continue
                obj_pts.append(X3)
                img_pts.append([float(detections[cam][f]["kps"][ki, 0]),
                                 float(detections[cam][f]["kps"][ki, 1])])

            n_corr = len(obj_pts)
            if n_corr < 6:
                continue   # not enough yet — try again after more landmarks

            obj_arr = np.array(obj_pts, dtype=np.float64)
            img_arr = np.array(img_pts, dtype=np.float64)
            K_c, d_c = cams[cam]["K"], cams[cam]["dist"]

            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj_arr, img_arr, K_c, d_c,
                iterationsCount=2000, reprojectionError=8.0,
                confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)

            n_in = len(inliers) if (ok and inliers is not None) else 0
            if not ok or n_in < 4:
                print(f"  cam {cam}: solvePnP failed  "
                      f"({n_corr} pts, {n_in} inliers) — deferred")
                continue

            R_c, _ = cv2.Rodrigues(rvec)
            T_c    = tvec.ravel()
            C_c    = (-R_c.T @ T_c)
            print(f"  cam {cam}: solvePnP  {n_corr} pts, {n_in} inliers  "
                  f"|T|={np.linalg.norm(T_c):.0f} mm  "
                  f"C=[{C_c[0]:.0f},{C_c[1]:.0f},{C_c[2]:.0f}]")
            poses[cam] = {"R": R_c, "T": T_c.reshape(3, 1)}
            placed.add(cam)
            placed_this_pass.append(cam)

            # Triangulate new landmarks using this camera + anchor ca
            # to expand coverage for cameras not yet placed
            P_new = _make_P(cam, poses[cam])
            for f in sorted(lm_frames):
                if f not in detections[cam] or f not in detections[ca]:
                    continue
                kps_new = detections[cam][f]["kps"]
                kps_ref = detections[ca][f]["kps"]
                for ki in SCALE_KPS:
                    if (f, ki) in landmarks:
                        continue
                    if (kps_new[ki, 2] < kp_conf_used or
                            kps_ref[ki, 2] < kp_conf_used):
                        continue
                    X3 = _tri(Pa, P_new,
                               float(kps_ref[ki, 0]), float(kps_ref[ki, 1]),
                               float(kps_new[ki, 0]), float(kps_new[ki, 1]))
                    if X3 is None:
                        continue
                    if (Ra @ X3 + Ta)[2] > 0 and (R_c @ X3 + T_c)[2] > 0:
                        landmarks[(f, ki)] = X3

            lm_frames = set(f for f, _ in landmarks)

        remaining = [c for c in remaining if c not in placed_this_pass]
        if not placed_this_pass:
            break   # no progress — remaining cameras can't be placed

    # Cameras that could never be placed
    failed = []
    for cam in remaining:
        print(f"  cam {cam}: could not place — identity pose (insufficient landmarks)")
        poses[cam] = {"R": np.eye(3, dtype=np.float64),
                      "T": np.zeros((3, 1), dtype=np.float64)}
        failed.append(cam)

    print(f"\n  Total landmarks: {len(landmarks)}  "
          f"(across {len(lm_frames)} frames)")
    print(f"  Cameras placed: {len(placed)}/{len(cam_ids_s)}  "
          f"  failed: {failed if failed else 'none'}")

    # ── 6. Bundle adjustment ──────────────────────────────────────────────────
    # Fix anchor ca at identity. Optimize rvec+tvec for all other cameras.
    # 3D landmarks are fixed (camera-only BA).
    opt_cams = [c for c in cam_ids_s if c != ca and c not in failed]
    n_opt    = len(opt_cams)

    # Build observation arrays
    obs_X3   = []   # (N, 3) world points
    obs_camI = []   # (N,)  index into opt_cams (-1 = anchor)
    obs_x    = []   # (N,)  observed u
    obs_y    = []   # (N,)  observed v

    for (f, ki), X3 in landmarks.items():
        for cam_id in cam_ids_s:
            det = detections[cam_id].get(f)
            if det is None or det["kps"][ki, 2] < 0.3:
                continue
            obs_X3.append(X3)
            obs_camI.append(opt_cams.index(cam_id) if cam_id in opt_cams else -1)
            obs_x.append(float(det["kps"][ki, 0]))
            obs_y.append(float(det["kps"][ki, 1]))

    obs_X3   = np.array(obs_X3,   dtype=np.float64)   # (N, 3)
    obs_camI = np.array(obs_camI, dtype=np.int32)
    obs_x    = np.array(obs_x,    dtype=np.float64)
    obs_y    = np.array(obs_y,    dtype=np.float64)
    N_obs    = len(obs_x)

    # Precompute per-camera K and dist arrays
    K_opt = [cams[c]["K"]    for c in opt_cams]
    d_opt = [cams[c]["dist"] for c in opt_cams]
    K_ca  = cams[ca]["K"]; d_ca = cams[ca]["dist"]

    # Masks per camera (precomputed for speed)
    masks = {i: (obs_camI == i) for i in range(n_opt)}
    mask_anchor = (obs_camI == -1)

    def _reproj_median(current_poses):
        errs = []
        for i, c in enumerate(opt_cams):
            msk = masks[i]
            if not msk.any():
                continue
            rv, _ = cv2.Rodrigues(current_poses[c]["R"])
            tv    = current_poses[c]["T"].ravel()
            proj  = _project_np(obs_X3[msk], rv, tv, K_opt[i], d_opt[i])
            errs.extend(np.hypot(proj[:, 0] - obs_x[msk],
                                  proj[:, 1] - obs_y[msk]).tolist())
        if mask_anchor.any():
            rv_a = np.zeros(3); tv_a = np.zeros(3)
            proj = _project_np(obs_X3[mask_anchor], rv_a, tv_a, K_ca, d_ca)
            errs.extend(np.hypot(proj[:, 0] - obs_x[mask_anchor],
                                  proj[:, 1] - obs_y[mask_anchor]).tolist())
        return float(np.median(errs)) if errs else 0.0

    reproj_before = _reproj_median(poses)
    print(f"\n  Bundle adjustment: {n_opt} cameras, {N_obs} observations")
    print(f"  Reprojection error BEFORE BA: {reproj_before:.1f} px (median)")

    # Pack initial parameters
    x0 = []
    for c in opt_cams:
        rv, _ = cv2.Rodrigues(poses[c]["R"])
        x0.extend(rv.ravel().tolist())
        x0.extend(poses[c]["T"].ravel().tolist())
    x0 = np.array(x0, dtype=np.float64)

    def residuals_fn(x):
        res = np.empty(N_obs * 2)
        for i in range(n_opt):
            msk = masks[i]
            if not msk.any():
                continue
            rv = x[i*6:i*6+3]; tv = x[i*6+3:i*6+6]
            proj = _project_np(obs_X3[msk], rv, tv, K_opt[i], d_opt[i])
            idx  = np.where(msk)[0]
            res[idx*2]     = proj[:, 0] - obs_x[msk]
            res[idx*2 + 1] = proj[:, 1] - obs_y[msk]
        if mask_anchor.any():
            rv_a = np.zeros(3); tv_a = np.zeros(3)
            proj = _project_np(obs_X3[mask_anchor], rv_a, tv_a, K_ca, d_ca)
            idx  = np.where(mask_anchor)[0]
            res[idx*2]     = proj[:, 0] - obs_x[mask_anchor]
            res[idx*2 + 1] = proj[:, 1] - obs_y[mask_anchor]
        return res

    result = least_squares(
        residuals_fn, x0,
        method='trf', loss='soft_l1', f_scale=10.0,
        max_nfev=300, verbose=0,
    )

    # Unpack BA result
    for i, c in enumerate(opt_cams):
        rv = result.x[i*6:i*6+3]
        tv = result.x[i*6+3:i*6+6]
        R_ba, _ = cv2.Rodrigues(rv)
        poses[c]["R"] = R_ba
        poses[c]["T"] = tv.reshape(3, 1)

    reproj_after = _reproj_median(poses)
    print(f"  Reprojection error AFTER  BA: {reproj_after:.1f} px (median)")
    print(f"  BA: converged={result.success}  cost={result.cost:.1f}  "
          f"nfev={result.nfev}")

    # ── 7. Finalise poses (P, C) ──────────────────────────────────────────────
    for cid in cam_ids_s:
        R = poses[cid]["R"]; T = poses[cid]["T"]
        poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R, T])
        poses[cid]["C"] = (-R.T @ T).ravel()

    # ── 8. Align room: origin at centroid(106,110), +X toward centroid(105,107) ──
    # Step 1: Translate so the centroid of cams 106 and 110 lands at viewer (0,0).
    #         Raw X = viewer X, raw Z = viewer Y — translate only these two axes
    #         so viewer Z (height, = -raw Y) is preserved.
    # Step 2: Rotate around raw Y (= viewer Z) so the direction from that origin
    #         toward the centroid of cams 105/107 aligns with viewer +X.
    #
    # Rotation of world frame by R_world (Ry in raw camera space):
    #   C_new = R_world @ C_translated
    #   R_new = R_old @ R_world^T
    #   T_new = -R_new @ C_new

    placed_ids = [c for c in cam_ids_s if c not in failed]

    origin_ids = [c for c in ["106", "110"] if c in placed_ids]
    xaxis_ids  = [c for c in ["105", "107"] if c in placed_ids]

    if len(origin_ids) < 1 or len(xaxis_ids) < 1:
        print(f"  [WARN] Room alignment skipped: "
              f"origin cams={origin_ids}  x-axis cams={xaxis_ids}")
    else:
        centroid_origin = np.mean([poses[c]["C"] for c in origin_ids], axis=0)
        centroid_xaxis  = np.mean([poses[c]["C"] for c in xaxis_ids],  axis=0)

        # Translate: centroid_origin moves to (0, y_unchanged, 0) in raw space
        t_align = np.array([-centroid_origin[0], 0.0, -centroid_origin[2]])

        # Direction in viewer XY (raw XZ) from translated origin to translated xaxis
        dx = float(centroid_xaxis[0] + t_align[0])
        dz = float(centroid_xaxis[2] + t_align[2])
        angle = np.arctan2(dz, dx)
        theta = -angle   # negate so that direction aligns with viewer +X

        c_t, s_t = float(np.cos(theta)), float(np.sin(theta))
        R_world = np.array([[c_t, 0., -s_t],
                             [0.,  1.,  0. ],
                             [s_t, 0.,  c_t]], dtype=np.float64)

        print(f"\n  Room alignment:")
        print(f"    Origin cameras : {origin_ids}  "
              f"centroid raw XZ = ({centroid_origin[0]:.1f}, {centroid_origin[2]:.1f}) mm")
        print(f"    X-axis cameras : {xaxis_ids}  "
              f"centroid raw XZ = ({centroid_xaxis[0]:.1f}, {centroid_xaxis[2]:.1f}) mm")
        print(f"    Yaw correction : {np.degrees(theta):.1f}°")

        for cid in cam_ids_s:
            if cid in failed:
                continue   # leave failed cameras at identity so viewer filter works
            C_old   = poses[cid]["C"]
            R_old   = poses[cid]["R"]
            C_trans = C_old + t_align
            C_new   = R_world @ C_trans
            R_new   = R_old @ R_world.T
            T_new   = (-R_new @ C_new).reshape(3, 1)
            poses[cid]["C"] = C_new
            poses[cid]["R"] = R_new
            poses[cid]["T"] = T_new
            poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R_new, T_new])

    save = {
        "reference_camera":      np.array([ca]),
        "reprojection_error_px": np.array([reproj_after]),
        "scale_factor":          np.array([scale]),
    }
    for cid in cam_ids_s:
        save[f"{cid}_R"] = poses[cid]["R"]
        save[f"{cid}_T"] = poses[cid]["T"]

    np.savez(str(poses_npz),  **save)
    np.savez(str(scaled_npz), **save)
    print(f"  Saved → {poses_npz}")
    print(f"  Saved → {scaled_npz}")

    return poses, ca, reproj_after


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
    """
    Write one ABT-compatible YAML per camera to PriCaB_output/yamls/.

    Each file contains:
      camera_matrix        — 3×3 intrinsic matrix K
      dist_coeffs          — 1×5 distortion coefficients [k1,k2,p1,p2,k3]
      rotation_matrix      — 3×3 extrinsic rotation R  (world → camera)
      translation_vector   — 3×1 translation T  (in mm)
      image_size           — [width, height] in pixels

    Files are always overwritten so they reflect the latest metric poses.
    """
    _header(6, "WRITE YAMLs  (ABT-compatible)")
    yaml_dir = OUT_DIR / "yamls"
    yaml_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in sorted(cam_ids):
        out = yaml_dir / f"{cam_id}.yaml"
        # Always overwrite — poses may have been updated by new extrinsics

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
    """
    Build a self-contained interactive HTML viewer.

    Layout
    ------
    Left half  — Plotly 3-D scatter (Z-up, floor = XY plane):
      - Red diamonds  : camera positions, labelled cam<id>
      - Blue circle   : animated hip midpoint (triangulated from ≥2 cameras)
      The 3-D scene is in the aligned world frame (mm):
        X  —  along the wall from cams 106/110 toward cams 105/107
        Y  —  into the room (perpendicular wall)
        Z  —  vertical (up)

    Right half — 2-column grid of camera feeds:
      - Frames are displayed in sync with the 3-D animation slider
      - YOLO bounding box overlaid in blue when a detection exists

    Cameras that could not be placed (identity pose, i.e. the 4 failed
    cameras 101/103/104/115 in the current dataset) are excluded from
    both panes automatically.

    Images are embedded as base64 JPEG (scale=IMG_SCALE, q=JPEG_QUALITY)
    making the file fully self-contained — no server or network needed.

    Skipped if pricab_viewer.html already exists; delete to force rebuild.

    Returns
    -------
    Path to the generated HTML file.
    """
    _header(7, "BUILD VIEWER  (self-contained HTML)")
    out_html = OUT_DIR / "pricab_viewer.html"
    if out_html.exists():
        print(f"  {out_html.name} already exists — skip")
        return out_html

    # Exclude cameras that failed placement (identity pose, non-anchor)
    cam_ids_s = [c for c in sorted(cam_ids)
                 if c == ref_cam
                 or not (np.allclose(poses[c]["R"], np.eye(3), atol=1e-6)
                         and np.allclose(poses[c]["T"], 0, atol=1e-3))]
    print(f"  Cameras in viewer: {cam_ids_s}  "
          f"(excluded: {sorted(set(cam_ids)-set(cam_ids_s))})")
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
        if X3 is not None and np.all(np.isfinite(X3)) and np.linalg.norm(X3) < 1e7:
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
    """
    Append calibration results to datalus_config.json.

    Writes / updates the following keys for each camera:
      position_mm      — X, Y, Z camera centre in the aligned world frame
      status           — "reference" for ref_cam, "calibrated" otherwise
      intrinsics_src   — string describing where intrinsics came from
      reprojection_px  — median reprojection error from bundle adjustment

    No-ops if datalus_config.json does not exist.
    """
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
