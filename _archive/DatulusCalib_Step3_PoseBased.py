#!/usr/bin/env python3
"""
Datalus Calibration — Step 3 (Pose-Based): Human Pose Extrinsic Calibration

Detects human body keypoints in the calibration frames using YOLOv8-pose
(or MediaPipe as fallback). Cameras are hardware-synchronized: frame N in
cam 102 equals frame N in cam 108.

Matched 2D keypoints are used to compute the Essential Matrix (RANSAC) and
recover relative camera poses — no checkerboard or 3D object points needed.

Steps:
1. Run pose detector on frames for each camera
2. For each camera pair, collect 2D-2D correspondences from synchronized frames
3. Compute Essential Matrix → recover R, T
4. Chain cameras through pose graph (cam 102 = reference, R=I, T=0)
5. Save colmap_poses.npz (same format as Steps 3 Stereo/RoomBased)

Run after DatulusCalib_Step2_Intrinsics.py.
"""

import os
import re
import sys
import cv2
import numpy as np
from pathlib import Path

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
    _f.write(f"\n{'='*40}\n{datetime.now():%Y-%m-%d %H:%M:%S}  "
             f"{Path(sys.argv[0]).name}\n{'='*40}\n")
    sys.stdout = _Tee(sys.__stdout__, _f)
    sys.stderr = _Tee(sys.__stderr__, _f)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FRAMES_ROOT    = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/frames"
INTRINSICS_NPZ = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/intrinsics.npz"
OUTPUT_DIR     = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration"

CAMERA_IDS     = ["102", "108", "113", "117"]
REFERENCE_CAM  = "108"

CONF_THRESHOLD     = 0.4   # minimum keypoint confidence to use
MIN_CORRESPONDENCES = 20   # minimum matched 2D points across all shared frames
MAX_FRAMES_PER_CAM = 600   # subsample to this many frames per camera

# COCO keypoint indices — stable landmarks visible from side and overhead views
# 0=nose, 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip, 13=L_knee, 14=R_knee
KEYPOINT_INDICES = [0, 5, 6, 11, 12, 13, 14]

# ─── DETECTOR ─────────────────────────────────────────────────────────────────

def load_detector():
    """Load YOLOv8-pose or MediaPipe, whichever is installed."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n-pose.pt")
        print("  Detector: YOLOv8n-pose (ultralytics)")
        return "yolo", model
    except Exception:
        pass
    try:
        import mediapipe as mp_lib
        mp_pose = mp_lib.solutions.pose
        model   = mp_pose.Pose(static_image_mode=True,
                               min_detection_confidence=0.4,
                               min_tracking_confidence=0.4)
        print("  Detector: MediaPipe Pose")
        return "mediapipe", model
    except Exception:
        pass
    print("[ERROR] No pose detector found. Install one of:")
    print("  pip install ultralytics")
    print("  pip install mediapipe")
    sys.exit(1)


# MediaPipe landmark index mapping for COCO keypoints used above
_COCO_TO_MP = {0: 0, 5: 11, 6: 12, 11: 23, 12: 24, 13: 25, 14: 26}


def _detect_yolo(model, img_path: Path):
    """Return list of (x,y)|None per KEYPOINT_INDICES, or None if no person."""
    results = model(str(img_path), verbose=False)
    if not results or results[0].keypoints is None:
        return None
    data = results[0].keypoints.data
    if data is None or data.shape[0] == 0:
        return None
    data = data.cpu().numpy()             # (n_persons, 17, 3)  [x, y, conf]
    best = int(np.argmax(data[:, :, 2].mean(axis=1)))
    pts  = data[best]                     # (17, 3)
    return [(float(pts[i, 0]), float(pts[i, 1]))
            if pts[i, 2] >= CONF_THRESHOLD else None
            for i in KEYPOINT_INDICES]


def _detect_mediapipe(model, img_path: Path):
    """Return list of (x,y)|None per KEYPOINT_INDICES, or None if no person."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    res  = model.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None
    lm = res.pose_landmarks.landmark
    out = []
    for i in KEYPOINT_INDICES:
        mp_i = _COCO_TO_MP.get(i)
        if mp_i is None or mp_i >= len(lm):
            out.append(None)
            continue
        p = lm[mp_i]
        out.append((p.x * w, p.y * h) if p.visibility >= CONF_THRESHOLD else None)
    return out


# ─── DETECTION LOOP ───────────────────────────────────────────────────────────

def detect_all(frames_root: str, camera_ids: list,
               detector_type: str, model) -> dict:
    """
    Returns dict: cam_id -> {(session, frame_idx): list_of_(x,y)|None}
    Only frames where at least 2 keypoints are detected are stored.
    """
    detect_fn = _detect_yolo if detector_type == "yolo" else _detect_mediapipe

    all_det = {}
    for cam_id in camera_ids:
        cam_dir = Path(frames_root) / cam_id
        images  = sorted(cam_dir.glob("*.png"))
        if not images:
            print(f"  [WARN] No frames for cam {cam_id}")
            all_det[cam_id] = {}
            continue

        # Subsample
        if len(images) > MAX_FRAMES_PER_CAM:
            idx_sel = np.linspace(0, len(images)-1, MAX_FRAMES_PER_CAM, dtype=int)
            images  = [images[i] for i in idx_sel]

        cam_det = {}
        n_det   = 0
        total   = len(images)

        for done, img_path in enumerate(images, 1):
            m = re.match(r"(\d+)_frame_(\d+)\.png", img_path.name)
            if m is None:
                continue
            key = (m.group(1), int(m.group(2)))
            kpts = detect_fn(model, img_path)
            if kpts is not None and sum(k is not None for k in kpts) >= 2:
                cam_det[key] = kpts
                n_det += 1
            fill = int(done / total * 25)
            sys.stdout.write(
                f"\r  cam {cam_id}: [{'#'*fill}{'-'*(25-fill)}] "
                f"{done}/{total}  detected={n_det}")
            sys.stdout.flush()
        print()
        all_det[cam_id] = cam_det
        print(f"    → {n_det} frames with person detected")

    return all_det


# ─── POSE ESTIMATION ──────────────────────────────────────────────────────────

def estimate_pose_pair(cam_a: str, cam_b: str,
                       det_a: dict, det_b: dict,
                       intrinsics: dict):
    """
    Estimate relative pose between cam_a and cam_b via Essential Matrix.
    Returns (R, T, n_inliers) where p_camB = R @ p_camA + T, or None.
    """
    shared = sorted(set(det_a.keys()) & set(det_b.keys()))
    if not shared:
        print(f"  [{cam_a}↔{cam_b}] no synchronized frames")
        return None

    Ka   = intrinsics[cam_a]["K"]
    da   = intrinsics[cam_a]["dist"]
    Kb   = intrinsics[cam_b]["K"]
    db   = intrinsics[cam_b]["dist"]

    pts_a, pts_b = [], []
    for key in shared:
        kpts_a = det_a[key]
        kpts_b = det_b[key]
        for j in range(len(KEYPOINT_INDICES)):
            if kpts_a[j] is not None and kpts_b[j] is not None:
                pts_a.append(kpts_a[j])
                pts_b.append(kpts_b[j])

    if len(pts_a) < MIN_CORRESPONDENCES:
        print(f"  [{cam_a}↔{cam_b}] only {len(pts_a)} correspondences "
              f"(need {MIN_CORRESPONDENCES}) — skipping")
        return None

    pts_a = np.array(pts_a, dtype=np.float64).reshape(-1, 1, 2)
    pts_b = np.array(pts_b, dtype=np.float64).reshape(-1, 1, 2)

    # Undistort and normalize to unit focal length
    pts_a_n = cv2.undistortPoints(pts_a, Ka, da)  # normalized (f=1, pp=0)
    pts_b_n = cv2.undistortPoints(pts_b, Kb, db)

    E, mask = cv2.findEssentialMat(
        pts_a_n, pts_b_n,
        focal=1.0, pp=(0.0, 0.0),
        method=cv2.RANSAC, prob=0.999, threshold=1e-3,
    )
    if E is None:
        print(f"  [{cam_a}↔{cam_b}] Essential matrix estimation failed")
        return None

    n_inliers, R, T, _ = cv2.recoverPose(E, pts_a_n, pts_b_n, mask=mask)
    print(f"  cam {cam_a} → cam {cam_b}:  "
          f"{len(pts_a)} correspondences  {n_inliers} inliers  "
          f"({len(shared)} shared frames)")
    return R, T.reshape(3, 1), n_inliers


# ─── POSE GRAPH ───────────────────────────────────────────────────────────────

def build_poses(camera_ids: list, reference_cam: str,
                detections: dict, intrinsics: dict) -> tuple:
    """
    Build absolute world-to-camera poses.
    Reference camera: R=I, T=0.
    Others: computed via Essential Matrix, chained through pose graph.
    Returns (abs_poses dict, cam_n_inliers dict).
    """
    abs_poses    = {reference_cam: (np.eye(3), np.zeros((3, 1)))}
    cam_n        = {reference_cam: 0}
    remaining    = [c for c in camera_ids if c != reference_cam]

    for _ in range(len(remaining) + 1):
        if not remaining:
            break
        for cam in list(remaining):
            for anchor, (R_anchor, T_anchor) in abs_poses.items():
                if anchor not in intrinsics or cam not in intrinsics:
                    continue
                result = estimate_pose_pair(
                    anchor, cam,
                    detections[anchor], detections[cam],
                    intrinsics,
                )
                if result is None:
                    continue
                R_rel, T_rel, n = result
                R_abs = R_rel @ R_anchor
                T_abs = R_rel @ T_anchor + T_rel
                abs_poses[cam] = (R_abs, T_abs)
                cam_n[cam]     = n
                remaining.remove(cam)
                break

    if remaining:
        print(f"  [WARN] Could not compute poses for: {remaining}")

    return abs_poses, cam_n


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _setup_log(str(Path(__file__).parent / "run.log"))
    print("=" * 60)
    print("DATALUS — STEP 3: POSE-BASED EXTRINSIC CALIBRATION")
    print("=" * 60)

    # ── Load intrinsics ────────────────────────────────────────────────────────
    if not os.path.exists(INTRINSICS_NPZ):
        print(f"[ERROR] {INTRINSICS_NPZ} not found. Run Step 2 first.")
        sys.exit(1)
    data       = np.load(INTRINSICS_NPZ)
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        if f"{cam_id}_K" in data:
            intrinsics[cam_id] = {
                "K":    data[f"{cam_id}_K"],
                "dist": data[f"{cam_id}_dist"],
            }
        else:
            print(f"  [WARN] No intrinsics for cam {cam_id}")

    print(f"\n  Reference camera : {REFERENCE_CAM}")
    print(f"  Frames root      : {FRAMES_ROOT}")
    print(f"  Max frames/cam   : {MAX_FRAMES_PER_CAM}")
    print(f"  Confidence thr.  : {CONF_THRESHOLD}")

    # ── Load detector ─────────────────────────────────────────────────────────
    print("\n[1] LOADING POSE DETECTOR")
    detector_type, model = load_detector()

    # ── Detect keypoints ──────────────────────────────────────────────────────
    print("\n[2] DETECTING KEYPOINTS")
    detections = detect_all(FRAMES_ROOT, CAMERA_IDS, detector_type, model)

    for cam_id in CAMERA_IDS:
        print(f"  cam {cam_id}: {len(detections.get(cam_id, {}))} frames with detections")

    # ── Build pose graph ──────────────────────────────────────────────────────
    print("\n[3] ESTIMATING CAMERA POSES")
    abs_poses, cam_n = build_poses(CAMERA_IDS, REFERENCE_CAM,
                                   detections, intrinsics)

    if len(abs_poses) < 2:
        print("[ERROR] Could not compute poses for enough cameras.")
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path  = Path(OUTPUT_DIR) / "colmap_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in abs_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
        save_dict[f"{cam_id}_n"] = np.array([cam_n.get(cam_id, 0)])
    np.savez(str(out_path), **save_dict)
    print(f"\n  Poses saved → {out_path}")

    print("\nSummary:")
    for cam_id, (R, T) in abs_poses.items():
        C = -R.T @ T
        n = cam_n.get(cam_id, 0)
        label = f"{n} inliers" if n > 0 else "reference"
        print(f"  cam {cam_id}: centre=[{C[0,0]:.4f}, {C[1,0]:.4f}, {C[2,0]:.4f}]"
              f"  ({label})")

    print("\nStep 3 complete.")
    print("Next: fill in world_registration.csv then run Step 4.")


if __name__ == "__main__":
    main()
