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
_REPO           = BASE_DIR.parent
INTRINSICS_NPZ  = _REPO / "_data" / "DatalusCalibration" / "intrinsics.npz"
CONFIG_JSON     = _REPO / "_archive" / "datalus_config.json"
POSE_MODEL_PATH = _REPO / "_models" / "yolo26x-pose.pt"
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

    # ── 9. Floor leveling: correct roll/pitch using co-planar cameras ────────────
    # Cameras 106, 107, 109, 112 are all mounted at the same height on the walls.
    # Fit a plane through them via SVD; rotate the world so that plane is horizontal.
    # This corrects any roll/pitch introduced by recoverPose (no gravity constraint).
    # Math: X_new = R_level @ X_old → R_new = R_old @ R_level^T, T_new = T_old.
    LEVEL_CAMS = [c for c in ["106", "107", "109", "112"] if c in placed_ids]
    if len(LEVEL_CAMS) >= 3:
        level_pts  = np.array([poses[c]["C"] for c in LEVEL_CAMS])
        centroid_l = level_pts.mean(axis=0)
        _, _, Vt   = np.linalg.svd(level_pts - centroid_l)
        n_current  = Vt[-1].astype(np.float64)        # floor normal

        # "up" in raw world ≈ (0,−1,0)  because viewer_vz = −raw_y
        n_target = np.array([0.0, -1.0, 0.0])
        if np.dot(n_current, n_target) < 0:
            n_current = -n_current

        v     = np.cross(n_current, n_target)
        s     = float(np.linalg.norm(v))
        c_dot = float(np.dot(n_current, n_target))
        tilt_deg = float(np.degrees(np.arcsin(min(s, 1.0))))

        print(f"\n  Floor leveling (pitch/roll correction):")
        print(f"    Co-planar cams : {LEVEL_CAMS}")
        print(f"    Tilt magnitude : {tilt_deg:.2f}°")

        if s > 1e-6:
            vx = np.array([[ 0,    -v[2],  v[1]],
                           [ v[2],  0,    -v[0]],
                           [-v[1],  v[0],  0   ]], dtype=np.float64)
            R_level = np.eye(3) + vx + vx @ vx * (1.0 - c_dot) / (s * s)

            h_before = [-poses[c]["C"][1] for c in LEVEL_CAMS]   # viewer_vz = -raw_y
            for cid in cam_ids_s:
                if cid in failed:
                    continue
                C_old = poses[cid]["C"]
                R_old = poses[cid]["R"]
                C_new = R_level @ C_old
                R_new = R_old @ R_level.T
                T_new = (-R_new @ C_new).reshape(3, 1)
                poses[cid]["C"] = C_new
                poses[cid]["R"] = R_new
                poses[cid]["T"] = T_new
                poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R_new, T_new])
            h_after = [-poses[c]["C"][1] for c in LEVEL_CAMS]
            print(f"    Height spread before : {max(h_before)-min(h_before):.0f} mm  "
                  f"(range {min(h_before):.0f} – {max(h_before):.0f})")
            print(f"    Height spread after  : {max(h_after)-min(h_after):.0f} mm  "
                  f"(range {min(h_after):.0f} – {max(h_after):.0f})")
        else:
            print("    Already level — skip")
    else:
        print(f"  Floor leveling: insufficient co-planar cams ({LEVEL_CAMS}) — skip")

    # ── 10. Chirality check: negate X if 106 is to the left of 107 ───────────────
    # recoverPose can return a chirality-flipped solution. We expect cam 106 to be
    # to the right of cam 107 (larger viewer-X), so if 106.C[0] < 107.C[0] we flip.
    if "106" in placed_ids and "107" in placed_ids:
        if poses["106"]["C"][0] < poses["107"]["C"][0]:
            print(f"\n  Chirality: negating X axis "
                  f"(cam106.x={poses['106']['C'][0]:.0f} < cam107.x={poses['107']['C'][0]:.0f})")
            mirror = np.diag([-1., 1., 1.])
            for cid in cam_ids_s:
                if cid in failed:
                    continue
                C_old = poses[cid]["C"]
                R_old = poses[cid]["R"]
                C_new = mirror @ C_old
                R_new = R_old @ mirror          # mirror is self-inverse
                T_new = (-R_new @ C_new).reshape(3, 1)
                poses[cid]["C"] = C_new
                poses[cid]["R"] = R_new
                poses[cid]["T"] = T_new
                poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R_new, T_new])
        else:
            print(f"\n  Chirality: OK (cam106.x={poses['106']['C'][0]:.0f} > cam107.x)")

    # Print final camera heights for reference
    print("\n  Final camera heights (viewer vz = -C[1]):")
    for cid in sorted([c for c in cam_ids_s if c not in failed]):
        C = poses[cid]["C"]
        vx, vy, vz = C[0], C[2], -C[1]
        print(f"    cam {cid}: height={vz:7.0f} mm  (vx={vx:7.0f}  vy={vy:7.0f})")

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


def _triangulate_all_keypoints(shared_frames, cam_ids_s, detections, poses):
    """
    Triangulate all 17 COCO keypoints for each frame.
    Returns {frame: [17 × [x,y,z] or None]}  in Z-up viewer coords.
    """
    result = {}
    for frame in shared_frames:
        kps_per_cam = {c: detections[c][frame]["kps"]
                       for c in cam_ids_s if frame in detections[c]}
        frame_kps = []
        for ki in range(17):
            obs = {}
            for cam_id, kps in kps_per_cam.items():
                x, y, c = kps[ki]
                if c >= KP_CONF_THRESH:
                    obs[cam_id] = (float(x), float(y))
            if len(obs) < 2:
                frame_kps.append(None)
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
                if (Ra @ X3 + Ta)[2] > 0:
                    pts3d.append(X3)
            if not pts3d:
                frame_kps.append(None)
                continue
            # Robust estimate: median → discard outliers >300mm → re-median
            arr = np.array(pts3d)
            med = np.median(arr, axis=0)
            if len(arr) >= 3:
                dists   = np.linalg.norm(arr - med, axis=1)
                inliers = arr[dists < 300.0]
                X3 = np.median(inliers, axis=0) if len(inliers) >= 1 else med
            else:
                X3 = med
            frame_kps.append([float(X3[0]), float(X3[2]), float(-X3[1])])
        result[frame] = frame_kps
    return result


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
                         frames_dirs, ref_cam, reproj_err=0.0, n_total_cams=0,
                         intrinsics_sources=None):
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

    # ── Triangulate all 17 keypoints ──────────────────────────────────────────
    kp3d = _triangulate_all_keypoints(shared_frames, cam_ids_s, detections, poses)
    n_full = sum(1 for f in kp3d if any(v is not None for v in kp3d[f]))
    print(f"  Full skeleton triangulated: {n_full}/{len(shared_frames)} frames")

    # ── Encode images ─────────────────────────────────────────────────────────
    print(f"  Encoding {len(shared_frames) * len(cam_ids_s)} images "
          f"(scale={IMG_SCALE}, q={JPEG_QUALITY})…")
    images = {c: [] for c in cam_ids_s}
    for frame in shared_frames:
        for c in cam_ids_s:
            img_path = frames_dirs[c] / f"{c}_frame_{frame:06d}.png"
            images[c].append(_encode_image(img_path, IMG_SCALE, JPEG_QUALITY))
    print("  Encoding done.")

    # ── Bounding boxes + keypoints for JS ────────────────────────────────────
    det_js = {}
    for frame in shared_frames:
        fd = {}
        for c in cam_ids_s:
            if frame in detections[c]:
                fd[c] = {
                    "bbox": detections[c][frame]["bbox"],
                    "kps":  detections[c][frame]["kps"].tolist(),
                }
        det_js[frame] = fd

    # ── Coordinate rotation: OpenCV cam space (Y-down, Z-forward) → Z-up world
    # Rx(-90°): (x, y, z) → (x, z, -y)  — floor becomes XY plane, Z points up
    def _to_zup(v):
        return [v[0], v[2], -v[1]]

    # ── Camera centres for 3D ─────────────────────────────────────────────────
    cam_centres = {c: _to_zup(poses[c]["C"].tolist()) for c in cam_ids_s}

    # ── Sort cameras by azimuthal angle around room centroid ──────────────────
    # This places physically-adjacent cameras next to each other in the strip,
    # so corner cameras end up at the extremes (left/right edges) of the strip.
    import math
    cx_floor = sum(cam_centres[c][0] for c in cam_ids_s) / len(cam_ids_s)
    cy_floor = sum(cam_centres[c][1] for c in cam_ids_s) / len(cam_ids_s)
    def _cam_angle(c):
        dx = cam_centres[c][0] - cx_floor
        dy = cam_centres[c][1] - cy_floor
        return math.atan2(dy, dx)
    cam_ids_s = sorted(cam_ids_s, key=_cam_angle)

    # ── Grid layout: all cameras in bottom strip, 2 rows ─────────────────────
    n_cams   = len(cam_ids_s)
    n_cols   = math.ceil(n_cams / 2)
    # Strip height sized so each cell matches native aspect ratio
    strip_h_css = (f"calc(100vw / {n_cols} * {native_h / native_w:.4f} * 2)")

    cam_divs = "\n      ".join(
        f'<div class="cam-wrap"><canvas id="c{c}"></canvas>'
        f'<div class="cam-label">cam {c}</div></div>'
        for c in cam_ids_s
    )
    canvas_js_obj = "{\n" + ",\n".join(
        f"  '{c}': document.getElementById('c{c}')" for c in cam_ids_s
    ) + "\n}"

    # ── Info panel static values ──────────────────────────────────────────────
    build_date  = datetime.now().strftime("%Y-%m-%d %H:%M")
    build_day   = datetime.now().strftime("%Y-%m-%d")
    scale_mm    = "N/A"
    _scaled_npz = OUT_DIR / "pricab_poses_scaled.npz"
    if _scaled_npz.exists():
        try:
            _d = np.load(_scaled_npz, allow_pickle=True)
            if "scale_factor" in _d:
                scale_mm = f"{float(_d['scale_factor'][0]):.4f}"
        except Exception:
            pass

    # Per-camera info for panel
    _src = intrinsics_sources or {}
    def _intr_tag(c):
        s = _src.get(c, "")
        return "calibrated" if "loaded" in s else "estimated"
    def _det_count(c):
        return len(detections.get(c, {}))
    # Anchor camera is the one whose pose is identity (ref_cam)
    _placed_ids = [c for c in cam_ids_s if c not in
                   [x for x in cam_ids if np.allclose(poses[x]["R"], np.eye(3), atol=1e-6)
                    and np.allclose(poses[x]["T"], 0, atol=1e-3) and x != ref_cam]]
    # Build HTML rows for per-camera table
    _cam_rows_html = ""
    for _c in sorted(cam_ids_s):
        _it = _intr_tag(_c)
        _it_color = "#44aadd" if _it == "calibrated" else "#ddaa44"
        _dc = _det_count(_c)
        _ref_mark = " ★" if _c == ref_cam else ""
        _cam_rows_html += (
            f'<tr><td class="tc">{_c}{_ref_mark}</td>'
            f'<td class="tc">{_dc}</td>'
            f'<td class="tc" style="color:{_it_color}">{_it[:3]}</td></tr>'
        )
    # Failed cameras
    _failed_ids = sorted(set(cam_ids) - set(cam_ids_s))
    _failed_str = ", ".join(_failed_ids) if _failed_ids else "none"
    # Video duration
    _dur_s = cam_info[cam_ids_s[0]]["duration_s"] if cam_ids_s else 0
    _fps_src = cam_info[cam_ids_s[0]]["fps"] if cam_ids_s else 0

    # ── JSON blobs ────────────────────────────────────────────────────────────
    j_frames      = json.dumps(shared_frames)
    j_images      = json.dumps(images)
    j_detections  = json.dumps(det_js)
    j_positions   = json.dumps({str(k): v for k, v in positions3d.items()})
    j_kp3d        = json.dumps({str(k): v for k, v in kp3d.items()})
    j_cam_ids     = json.dumps(cam_ids_s)
    j_img_dims    = json.dumps({"native_w": native_w, "native_h": native_h,
                                 "disp_w": disp_w, "disp_h": disp_h})
    j_cam_centres = json.dumps({c: cam_centres[c] for c in cam_ids_s})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PriCaB — Human Calibration Viewer</title>
<script type="importmap">
{{"imports": {{
  "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
  "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}}}}
</script>
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
#bottom-cams {{
  height: {strip_h_css};
  flex-shrink: 0;
  display: grid;
  grid-template-columns: repeat({n_cols}, 1fr);
  grid-template-rows: 1fr 1fr;
  gap: 2px; background: #050510; padding: 2px;
  border-top: 1px solid #1a2a3a;
}}
/* shared camera cell styles */
.cam-wrap {{ position: relative; background: #080818; overflow: hidden; }}
.cam-wrap canvas {{ width: 100%; height: 100%; display: block; }}
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
@keyframes pulse-rec {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:0.5; }} }}
#slider {{ flex: 1; accent-color: #4477cc; height: 4px; cursor: pointer; }}
#frame-label {{ min-width: 140px; text-align: right; font-size: 12px; color: #7799bb; }}
</style>
</head>
<body>
<div id="header" style="display:flex;justify-content:space-between;align-items:center;">
  <div>
    <span style="color:#ccd8ee;font-size:14px;font-weight:bold;">PriCaB &mdash; Primate Cognition and Behavior</span>
    <span style="color:#445566">&nbsp;&nbsp;/&nbsp;&nbsp;</span>
    <span style="color:#7799bb">Multi-camera 3D Pose Viewer</span>
    <span style="color:#334455;margin-left:16px;font-size:11px;">
      <span style="color:#ff6655">&#9670;</span> cameras &nbsp;
      <span style="color:#44aaff">&#9135;</span> CoM disc &nbsp;
      <span style="color:#44aaff">&#11044;</span> keypoints &nbsp;&nbsp;
      drag&nbsp;rotate &middot; scroll&nbsp;zoom &middot; right-drag&nbsp;pan
    </span>
  </div>
  <div style="text-align:right;font-size:11px;color:#445566;white-space:nowrap;">
    <span style="color:#3a6090">acalapai@dpz.eu</span>&nbsp;&nbsp;|&nbsp;&nbsp;{build_day}
  </div>
</div>
<div id="main">
  <div id="viewer-row">
    <div id="info-panel">

      <div class="info-section">
        <h2>Recording</h2>
        <div class="info-row"><span class="info-key">Session</span><span class="info-val" style="font-size:10px;color:#7799aa;">{Path(frames_dirs[cam_ids_s[0]]).parent.parent.name if cam_ids_s else "—"}</span></div>
        <div class="info-row"><span class="info-key">Video&nbsp;fps</span><span class="info-val">{_fps_src:.1f}&thinsp;Hz</span></div>
        <div class="info-row"><span class="info-key">Duration</span><span class="info-val">{_dur_s:.1f}&thinsp;s</span></div>
        <div class="info-row"><span class="info-key">Sensor</span><span class="info-val">{native_w}&thinsp;&times;&thinsp;{native_h}</span></div>
        <div class="info-row"><span class="info-key">Cameras&nbsp;total</span><span class="info-val">{n_total_cams}</span></div>
        <div class="info-row"><span class="info-key">Placed</span><span class="info-val">{len(cam_ids_s)}</span></div>
        <div class="info-row"><span class="info-key">Failed</span><span class="info-val" style="color:#dd7744">{_failed_str}</span></div>
      </div>

      <div class="info-section">
        <h2>Pose Detection</h2>
        <div class="info-row"><span class="info-key">Model</span><span class="info-val">yolov8n-pose</span></div>
        <div class="info-row"><span class="info-key">Keypoints</span><span class="info-val">COCO&thinsp;17</span></div>
        <div class="info-row"><span class="info-key">Sample&nbsp;rate</span><span class="info-val">500&thinsp;ms</span></div>
        <div class="info-row"><span class="info-key">KP&nbsp;conf&nbsp;thr</span><span class="info-val">0.5</span></div>
        <div class="info-row"><span class="info-key">BBox&nbsp;conf&nbsp;thr</span><span class="info-val">0.3</span></div>
        <div class="info-row"><span class="info-key">Frames&nbsp;used</span><span class="info-val">{len(shared_frames)}</span></div>
      </div>

      <div class="info-section">
        <h2>Calibration</h2>
        <div class="info-row"><span class="info-key">Method</span><span class="info-val" style="font-size:10px;">essmat&thinsp;+&thinsp;solvePnP&thinsp;+&thinsp;BA</span></div>
        <div class="info-row"><span class="info-key">Reference&nbsp;cam</span><span class="info-val">{ref_cam}</span></div>
        <div class="info-row"><span class="info-key">Reproj&nbsp;error</span><span class="info-val">{reproj_err:.1f}&thinsp;px</span></div>
        <div class="info-row"><span class="info-key">Person&nbsp;height</span><span class="info-val">1700&thinsp;mm</span></div>
        <div class="info-row"><span class="info-key">Scale&nbsp;factor</span><span class="info-val">{scale_mm}</span></div>
        <div class="info-row"><span class="info-key">Floor&nbsp;leveling</span><span class="info-val">SVD&thinsp;plane</span></div>
        <div class="info-row"><span class="info-key">Yaw&nbsp;align</span><span class="info-val">cam&thinsp;106/110&thinsp;axis</span></div>
      </div>

      <div class="info-section">
        <h2>3D Reconstruction</h2>
        <div class="info-row"><span class="info-key">Method</span><span class="info-val" style="font-size:10px;">DLT&thinsp;triangulation</span></div>
        <div class="info-row"><span class="info-key">Inlier&nbsp;thr</span><span class="info-val">300&thinsp;mm</span></div>
        <div class="info-row"><span class="info-key">Bone&nbsp;max</span><span class="info-val">700&thinsp;mm</span></div>
        <div class="info-row"><span class="info-key">Hip&nbsp;triangulated</span><span class="info-val">{len(positions3d)}&thinsp;/&thinsp;{len(shared_frames)}</span></div>
        <div class="info-row"><span class="info-key">Full&nbsp;skeleton</span><span class="info-val">{n_full}&thinsp;/&thinsp;{len(shared_frames)}</span></div>
      </div>

      <div class="info-section">
        <h2>Cameras</h2>
        <table class="cam-table">
          <tr><th>ID</th><th>det.</th><th>K</th></tr>
          {_cam_rows_html}
        </table>
        <div style="font-size:9px;color:#3a5570;margin-top:3px;">
          ★ reference &nbsp; K: cal=calibrated, est=estimated
        </div>
      </div>

      <div class="info-section">
        <h2>Live</h2>
        <div class="info-row"><span class="info-key">Frame</span><span class="info-val live" id="li-frame">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Hip&nbsp;X</span><span class="info-val live" id="li-hx">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Hip&nbsp;Y</span><span class="info-val live" id="li-hy">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Hip&nbsp;Z</span><span class="info-val live" id="li-hz">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Cams&nbsp;detect.</span><span class="info-val live" id="li-ncams">&mdash;</span></div>
        <div class="info-row"><span class="info-key">Visible&nbsp;KPs</span><span class="info-val live" id="li-nkps">&mdash;</span></div>
      </div>

      <div class="info-section" style="margin-top:auto;padding-top:8px;border-top:1px solid #1a2a3a;font-size:10px;color:#3a5570;line-height:1.6;">
        Primate Cognition and Behavior Lab<br>
        Deutsches Primatenzentrum<br>
        G&ouml;ttingen, Germany<br>
        <span style="color:#2a5080">acalapai@dpz.eu</span>
      </div>

    </div>
    <div id="plot-panel"></div>
  </div>
  <div id="bottom-cams">
    {cam_divs}
  </div>
</div>
<div id="controls">
  <div id="ctrl-row">
    <button class="btn" id="btnPlay">&#9654; Play</button>
    <button class="btn" id="btnPause">&#9646;&#9646; Pause</button>
    <button class="btn active" id="btnToggle">&#9135; Bones</button>
    <button class="btn" id="btnRec">&#9210; Rec</button>
    <label style="color:#7799bb;font-size:12px;">Speed:
      <select id="speedSel" style="background:#1a2a4a;color:#aaccff;border:1px solid #335;border-radius:4px;padding:2px 6px;font-family:monospace;">
        <option value="500">0.4×</option>
        <option value="333">0.6×</option>
        <option value="200" selected>1×</option>
        <option value="133">1.5×</option>
        <option value="100">2×</option>
      </select>
    </label>
    <input type="range" id="slider" min="0" max="{len(shared_frames)-1}" value="0">
    <span id="frame-label">frame {shared_frames[0] if shared_frames else 0}</span>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

// ── Data ──────────────────────────────────────────────────────────────────────
const FRAMES      = {j_frames};
const IMAGES      = {j_images};
const DETECTIONS  = {j_detections};
const POSITIONS   = {j_positions};
const KP3D        = {j_kp3d};
const IMG_DIMS    = {j_img_dims};
const CAM_IDS     = {j_cam_ids};
const CAM_CENTRES = {j_cam_centres};

const COCO_SKEL = [
  [15,13],[13,11],[16,14],[14,12],[11,12],
  [5,11],[6,12],[5,6],[5,7],[6,8],[7,9],[8,10],
  [1,2],[0,1],[0,2],[1,3],[2,4],[3,5],[4,6]
];
const KP_COLORS_HEX = [
  0xff6b6b,0xff9f43,0xffd32a,0xc4e538,0x67e217,
  0x1dd1a1,0x00d2d3,0x54a0ff,0x5f27cd,0x341f97,
  0xff9f43,0xee5a24,0x009432,0x0652dd,0x1289a7,
  0xc4e538,0xed4c67
];
const KP_COLORS_CSS = [
  '#ff6b6b','#ff9f43','#ffd32a','#c4e538','#67e217',
  '#1dd1a1','#00d2d3','#54a0ff','#5f27cd','#341f97',
  '#ff9f43','#ee5a24','#009432','#0652DD','#1289A7',
  '#C4E538','#ED4C67'
];

// ── Three.js scene ────────────────────────────────────────────────────────────
// Viewer coords: [vx, vy, vz] where vz = up.
// Three.js Y-up: map vx→x, vz→y, vy→z
function toThree(v) {{ return new THREE.Vector3(v[0], v[2], v[1]); }}

const panel = document.getElementById('plot-panel');
const W = panel.clientWidth, H = panel.clientHeight;

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W, H);
renderer.setClearColor(0x080818);
panel.appendChild(renderer.domElement);
renderer.domElement.id = 'three-canvas';

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(W, H);
labelRenderer.domElement.style.position = 'absolute';
labelRenderer.domElement.style.top = '0';
labelRenderer.domElement.style.pointerEvents = 'none';
panel.appendChild(labelRenderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x080818, 30000, 80000);

const camera = new THREE.PerspectiveCamera(55, W / H, 10, 200000);
camera.position.set(8000, 6000, 14000);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
// Orbit target = centroid of camera centres (computed after CAM_CENTRES loads)
{{
  const cc = Object.values(CAM_CENTRES);
  const cx = cc.reduce((s,c) => s+c[0], 0) / cc.length;
  const cy = cc.reduce((s,c) => s+c[1], 0) / cc.length;
  const cz = cc.reduce((s,c) => s+c[2], 0) / cc.length;
  controls.target.set(cx, cz, cy);   // toThree mapping: x→x, vz→y, vy→z
}}
controls.update();

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const sun = new THREE.DirectionalLight(0xffffff, 0.9);
sun.position.set(5000, 10000, 4000);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8899cc, 0.3);
fill.position.set(-4000, 2000, -3000);
scene.add(fill);

// Floor: 1000 mm below the lowest of cameras 106/107/109/112
const _lowerIds = ["106","107","109","112"].filter(c => CAM_CENTRES[c]);
const floorY = (_lowerIds.length > 0
  ? Math.min(..._lowerIds.map(c => CAM_CENTRES[c][2])) - 1000
  : -1500);

const gridHelper = new THREE.GridHelper(28000, 28, 0x1a2a4a, 0x0d1525);
gridHelper.position.y = floorY;
scene.add(gridHelper);

// ── Room walls (snapped to corner-camera positions, no margin) ────────────────
{{
  const cc  = Object.values(CAM_CENTRES);
  let xMin = Infinity, xMax = -Infinity;
  let yMin = Infinity, yMax = -Infinity;
  let zMax = -Infinity;
  for (const c of cc) {{
    xMin = Math.min(xMin, c[0]); xMax = Math.max(xMax, c[0]);
    yMin = Math.min(yMin, c[1]); yMax = Math.max(yMax, c[1]);
    zMax = Math.max(zMax, c[2]);
  }}
  // Walls pass exactly through the outermost (corner) cameras.
  // Add a small ceiling gap above the highest camera for visual clarity.
  const ceilY  = zMax + 400;
  const roomW  = xMax - xMin;
  const roomD  = yMax - yMin;
  const roomH  = ceilY - floorY;
  const midX   = (xMin + xMax) / 2;
  const midZ   = (yMin + yMax) / 2;
  const midY   = (floorY + ceilY) / 2;

  const wallMat = new THREE.MeshPhongMaterial({{
    color: 0x2244aa, emissive: 0x0a1133,
    opacity: 0.10, transparent: true,
    side: THREE.DoubleSide, depthWrite: false
  }});
  const edgeMat = new THREE.LineBasicMaterial({{
    color: 0x3366cc, opacity: 0.35, transparent: true
  }});

  function addPlane(w, h, px, py, pz, rx, ry, rz) {{
    const geo  = new THREE.PlaneGeometry(w, h);
    const mesh = new THREE.Mesh(geo, wallMat);
    mesh.position.set(px, py, pz);
    mesh.rotation.set(rx, ry, rz);
    scene.add(mesh);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
    edges.position.copy(mesh.position);
    edges.rotation.copy(mesh.rotation);
    scene.add(edges);
  }}

  addPlane(roomW, roomD,  midX,  floorY, midZ,  -Math.PI/2, 0, 0);  // floor
  addPlane(roomW, roomD,  midX,  ceilY,  midZ,   Math.PI/2, 0, 0);  // ceiling
  addPlane(roomD, roomH,  xMin,  midY,   midZ,   0,  Math.PI/2, 0); // -X wall
  addPlane(roomD, roomH,  xMax,  midY,   midZ,   0, -Math.PI/2, 0); // +X wall
  addPlane(roomW, roomH,  midX,  midY,   yMin,   0, 0, 0);          // -Z wall
  addPlane(roomW, roomH,  midX,  midY,   yMax,   0, Math.PI, 0);    // +Z wall
}}

// ── Camera markers ────────────────────────────────────────────────────────────
const camGeo = new THREE.OctahedronGeometry(140, 0);
const camMat = new THREE.MeshPhongMaterial({{ color: 0xff4444, emissive: 0x441111, shininess: 80 }});
for (const [id, c] of Object.entries(CAM_CENTRES)) {{
  const mesh = new THREE.Mesh(camGeo, camMat);
  mesh.position.copy(toThree(c));
  scene.add(mesh);

  const div = document.createElement('div');
  div.className = 'cam3d-label';
  div.textContent = 'cam ' + id;
  const lbl = new CSS2DObject(div);
  lbl.position.copy(toThree(c));
  lbl.position.y += 220;
  scene.add(lbl);
}}

// ── CoM sphere (semitransparent) ──────────────────────────────────────────────
const comGroup = new THREE.Group();
const _comMesh = new THREE.Mesh(
  new THREE.SphereGeometry(220, 24, 16),
  new THREE.MeshPhongMaterial({{
    color: 0x44aaff, emissive: 0x0a2244,
    opacity: 0.30, transparent: true, shininess: 80, depthWrite: false
  }})
);
comGroup.add(_comMesh);
scene.add(comGroup);

// ── Keypoint spheres ─────────────────────────────────────────────────────────
// Head (kps 0-4: nose, eyes, ears): replaced by a single head-centroid sphere.
// Body (kps 5-16): individual colored spheres.
const KP_RADIUS = 45;
const kpMeshes = KP_COLORS_HEX.map((col, i) => {{
  // Suppress individual head joints (0-4) — they jitter; we use headMesh instead
  if (i < 5) return null;
  const m = new THREE.Mesh(
    new THREE.SphereGeometry(KP_RADIUS, 10, 10),
    new THREE.MeshPhongMaterial({{ color: col, emissive: col, emissiveIntensity: 0.25, shininess: 90 }})
  );
  m.visible = false;
  scene.add(m);
  return m;
}});

// Head centroid sphere (represents CoM of visible head keypoints 0-4)
const headMesh = new THREE.Mesh(
  new THREE.SphereGeometry(KP_RADIUS * 1.4, 12, 12),
  new THREE.MeshPhongMaterial({{ color: 0xffd32a, emissive: 0x443300, emissiveIntensity: 0.3, shininess: 90 }})
);
headMesh.visible = false;
scene.add(headMesh);

// ── Bone cylinders (thick) ────────────────────────────────────────────────────
const BONE_RADIUS = 38;
const boneMat = new THREE.MeshPhongMaterial({{
  color: 0xffffff, emissive: 0x334455, opacity: 0.82, transparent: true, shininess: 60
}});
const boneMeshes = COCO_SKEL.map(() => {{
  const m = new THREE.Mesh(new THREE.CylinderGeometry(BONE_RADIUS, BONE_RADIUS, 1, 8), boneMat);
  m.visible = false;
  scene.add(m);
  return m;
}});

const _yAxis = new THREE.Vector3(0, 1, 0);
const _dir   = new THREE.Vector3();
const _mid   = new THREE.Vector3();
const _pA    = new THREE.Vector3();
const _pB    = new THREE.Vector3();

const MAX_BONE_MM = 700;  // biological plausibility threshold (mm)

// Neighbor map: for each kp index, which other kps are connected in COCO_SKEL
const KP_NEIGHBORS = Array.from({{length: 17}}, () => []);
for (const [a, b] of COCO_SKEL) {{
  KP_NEIGHBORS[a].push(b);
  KP_NEIGHBORS[b].push(a);
}}

function kpDist(kps, i, j) {{
  const a = kps[i], b = kps[j];
  if (!a || !b) return Infinity;
  return Math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2);
}}

// A keypoint is plausible only if at least one COCO neighbor is within MAX_BONE_MM
function isKpPlausible(kps, i) {{
  if (!kps[i]) return false;
  return KP_NEIGHBORS[i].some(j => kpDist(kps, i, j) <= MAX_BONE_MM);
}}

function placeBone(mesh, va, vb) {{
  _pA.set(va[0], va[2], va[1]);
  _pB.set(vb[0], vb[2], vb[1]);
  _dir.subVectors(_pB, _pA);
  const len = _dir.length();
  if (len < 1 || len > MAX_BONE_MM) {{ mesh.visible = false; return; }}
  _mid.addVectors(_pA, _pB).multiplyScalar(0.5);
  mesh.position.copy(_mid);
  mesh.scale.set(1, len, 1);
  mesh.quaternion.setFromUnitVectors(_yAxis, _dir.normalize());
  mesh.visible = true;
}}

// ── 3D update ─────────────────────────────────────────────────────────────────
// CoM disc and skeleton are always shown together.
// The toggle button controls bones visibility only.
let showBones = true;

function update3D(frameIdx) {{
  const frame = FRAMES[frameIdx].toString();
  const kps   = KP3D[frame] || [];

  // ── CoM disc (hip midpoint) ─────────────────────────────────────────────────
  const pos = POSITIONS[frame];
  if (pos) {{
    comGroup.position.set(pos[0], pos[2], pos[1]);
    comGroup.visible = true;
  }} else {{
    comGroup.visible = false;
  }}

  // ── Body keypoints (5-16) ───────────────────────────────────────────────────
  for (let i = 5; i < 17; i++) {{
    const m = kpMeshes[i];
    if (!m) continue;
    if (isKpPlausible(kps, i)) {{
      m.position.set(kps[i][0], kps[i][2], kps[i][1]);
      m.visible = true;
    }} else {{
      m.visible = false;
    }}
  }}

  // ── Head centroid ───────────────────────────────────────────────────────────
  const headPts = [0,1,2,3,4].filter(i => kps[i]).map(i => kps[i]);
  if (headPts.length > 0) {{
    const hx = headPts.reduce((s,p) => s+p[0], 0) / headPts.length;
    const hy = headPts.reduce((s,p) => s+p[1], 0) / headPts.length;
    const hz = headPts.reduce((s,p) => s+p[2], 0) / headPts.length;
    const nearShoulder = [5, 6].some(si => {{
      if (!kps[si]) return false;
      const dx=hx-kps[si][0], dy=hy-kps[si][1], dz=hz-kps[si][2];
      return Math.sqrt(dx*dx+dy*dy+dz*dz) <= MAX_BONE_MM;
    }});
    if (nearShoulder) {{
      headMesh.position.set(hx, hz, hy);
      headMesh.visible = true;
    }} else {{
      headMesh.visible = false;
    }}
  }} else {{
    headMesh.visible = false;
  }}

  // ── Bones (toggled by button) ───────────────────────────────────────────────
  if (showBones) {{
    COCO_SKEL.forEach(([a, b], bi) => {{
      if (isKpPlausible(kps, a) && isKpPlausible(kps, b))
        placeBone(boneMeshes[bi], kps[a], kps[b]);
      else boneMeshes[bi].visible = false;
    }});
  }} else {{
    boneMeshes.forEach(m => {{ m.visible = false; }});
  }}
}}

// ── Animation loop ────────────────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
  labelRenderer.render(scene, camera);
}}
animate();

// ── Camera feed canvases ──────────────────────────────────────────────────────
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

const COCO_SKELETON = COCO_SKEL;

function drawCanvas(camId, frameIdx) {{
  const canvas = canvases[camId];
  if (!canvas) return;
  const ctx  = canvas.getContext('2d');
  const dw   = canvas.offsetWidth  || IMG_DIMS.disp_w;
  const dh   = canvas.offsetHeight || IMG_DIMS.disp_h;
  canvas.width  = dw;
  canvas.height = dh;
  const img = getImg(camId, frameIdx);

  const imgAspect = IMG_DIMS.native_w / IMG_DIMS.native_h;
  const canAspect = dw / dh;
  let drawW, drawH, ox, oy;
  if (canAspect > imgAspect) {{
    drawH = dh; drawW = dh * imgAspect; ox = (dw - drawW) / 2; oy = 0;
  }} else {{
    drawW = dw; drawH = dw / imgAspect; ox = 0; oy = (dh - drawH) / 2;
  }}
  const scaleX = drawW / IMG_DIMS.native_w;
  const scaleY = drawH / IMG_DIMS.native_h;

  function draw() {{
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(img, ox, oy, drawW, drawH);
    const frame = FRAMES[frameIdx];
    const det   = (DETECTIONS[frame] || {{}})[camId];
    if (!det) return;
    const bbox = det.bbox, kps = det.kps;
    const lw = Math.max(1.5, dw * 0.002);
    const x1 = ox + bbox[0]*scaleX, y1 = oy + bbox[1]*scaleY;
    const x2 = ox + bbox[2]*scaleX, y2 = oy + bbox[3]*scaleY;
    ctx.strokeStyle = '#44aaff'; ctx.lineWidth = lw * 1.5;
    ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    ctx.fillStyle = '#44aaff';
    ctx.font = Math.round(dw * 0.032) + 'px monospace';
    ctx.fillText('person', x1+3, Math.max(y1-4, 14));
    ctx.lineWidth = lw;
    for (const [a, b] of COCO_SKELETON) {{
      if (kps[a][2] < 0.3 || kps[b][2] < 0.3) continue;
      ctx.beginPath(); ctx.strokeStyle = 'rgba(255,255,255,0.55)';
      ctx.moveTo(ox + kps[a][0]*scaleX, oy + kps[a][1]*scaleY);
      ctx.lineTo(ox + kps[b][0]*scaleX, oy + kps[b][1]*scaleY);
      ctx.stroke();
    }}
    const r = Math.max(2.5, dw * 0.006);
    for (let i = 0; i < 17; i++) {{
      if (kps[i][2] < 0.3) continue;
      ctx.beginPath();
      ctx.arc(ox + kps[i][0]*scaleX, oy + kps[i][1]*scaleY, r, 0, 2*Math.PI);
      ctx.fillStyle = KP_COLORS_CSS[i]; ctx.fill();
      ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.lineWidth = 1; ctx.stroke();
    }}
  }}
  if (img.complete) draw(); else img.onload = draw;
}}

// ── Controls ──────────────────────────────────────────────────────────────────
function setFrame(idx) {{
  CAM_IDS.forEach(c => drawCanvas(c, idx));
  update3D(idx);
  document.getElementById('frame-label').textContent =
    'frame ' + FRAMES[idx] + ' (' + (idx+1) + '/' + FRAMES.length + ')';
  document.getElementById('slider').value = idx;
  for (let i = idx+1; i < Math.min(idx+4, FRAMES.length); i++)
    CAM_IDS.forEach(c => getImg(c, i));

  // ── Update info panel live fields ──────────────────────────────────────────
  document.getElementById('li-frame').textContent = (idx+1) + ' / ' + FRAMES.length;
  const pos = POSITIONS[FRAMES[idx]];
  if (pos) {{
    document.getElementById('li-hx').textContent = pos[0].toFixed(0) + ' mm';
    document.getElementById('li-hy').textContent = pos[1].toFixed(0) + ' mm';
    document.getElementById('li-hz').textContent = pos[2].toFixed(0) + ' mm';
  }} else {{
    ['li-hx','li-hy','li-hz'].forEach(id =>
      document.getElementById(id).textContent = '—');
  }}
  const det = DETECTIONS[FRAMES[idx]];
  const ndet = det ? Object.keys(det).length : 0;
  document.getElementById('li-ncams').textContent = ndet + ' / ' + CAM_IDS.length;
  const kps = KP3D[FRAMES[idx]];
  const nkps = kps ? kps.filter(k => k !== null).length : 0;
  document.getElementById('li-nkps').textContent = nkps + ' / 17';
}}

const slider = document.getElementById('slider');
slider.addEventListener('input', () => setFrame(+slider.value));

let playTimer = null, playIdx = 0;
let fps_interval = 200;
document.getElementById('speedSel').addEventListener('change', e => {{
  fps_interval = +e.target.value;
  if (playTimer) {{   // restart with new speed if playing
    clearInterval(playTimer); playTimer = null;
    document.getElementById('btnPlay').click();
  }}
}});
document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playTimer) return;
  if (playIdx >= FRAMES.length - 1) playIdx = 0;
  playTimer = setInterval(() => {{
    playIdx++;
    setFrame(playIdx);
    if (playIdx >= FRAMES.length - 1) {{ clearInterval(playTimer); playTimer = null; }}
  }}, fps_interval);
}});
document.getElementById('btnPause').addEventListener('click', () => {{
  clearInterval(playTimer); playTimer = null;
  playIdx = +slider.value;
}});

const btnToggle = document.getElementById('btnToggle');
btnToggle.addEventListener('click', () => {{
  showBones = !showBones;
  btnToggle.classList.toggle('active', showBones);
  setFrame(+slider.value);
}});

window.addEventListener('resize', () => {{
  const pw = panel.clientWidth, ph = panel.clientHeight;
  camera.aspect = pw / ph;
  camera.updateProjectionMatrix();
  renderer.setSize(pw, ph);
  labelRenderer.setSize(pw, ph);
  setFrame(+slider.value);
}});

setFrame(0);

// ── Screen recording ──────────────────────────────────────────────────────────
{{
  const btnRec = document.getElementById('btnRec');
  let mediaRecorder = null;
  let recChunks = [];

  btnRec.addEventListener('click', async () => {{
    if (mediaRecorder && mediaRecorder.state === 'recording') {{
      // ── Stop ──────────────────────────────────────────────────────────────
      mediaRecorder.stop();
    }} else {{
      // ── Start ─────────────────────────────────────────────────────────────
      let stream;
      try {{
        stream = await navigator.mediaDevices.getDisplayMedia({{
          video: {{ frameRate: 30 }},
          audio: false
        }});
      }} catch (e) {{
        console.warn('Screen capture cancelled or unavailable:', e);
        return;
      }}

      // WebM is universally supported in Chrome; Safari supports mp4
      const mimeTypes = [
        'video/webm;codecs=vp9',
        'video/webm;codecs=vp8',
        'video/webm',
        'video/mp4;codecs=h264',
        'video/mp4',
      ];
      const mime = mimeTypes.find(t => MediaRecorder.isTypeSupported(t)) || '';

      recChunks = [];
      mediaRecorder = new MediaRecorder(stream, mime ? {{ mimeType: mime }} : {{}});

      mediaRecorder.ondataavailable = e => {{ if (e.data.size > 0) recChunks.push(e.data); }};

      mediaRecorder.onstop = () => {{
        stream.getTracks().forEach(t => t.stop());
        btnRec.textContent = '\u2609 Rec';
        btnRec.classList.remove('recording');
        mediaRecorder = null;

        if (recChunks.length === 0) return;  // cancelled before any data

        const ext  = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
        const ts   = new Date().toISOString().replace(/[:.]/g,'-').slice(0,19);
        const blob = new Blob(recChunks, {{ type: mime || 'video/webm' }});
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href = url; a.download = `pricab_recording_${{ts}}.${{ext}}`;
        a.click();
        URL.revokeObjectURL(url);
      }};

      // If user stops sharing via browser UI, honour it
      stream.getVideoTracks()[0].onended = () => {{
        if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
      }};

      mediaRecorder.start(1000);
      btnRec.textContent = '\u25a0 Stop';
      btnRec.classList.add('recording');
    }}
  }});
}}
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
        cam_ids, cam_info, detections, cams, poses, frames_dirs, ref_cam,
        reproj_err=reproj_err, n_total_cams=len(cam_ids),
        intrinsics_sources=intrinsics_sources)

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
