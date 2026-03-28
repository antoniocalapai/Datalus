#!/usr/bin/env python3
"""
Datalus Calibration — Step 3: Pose-Correspondence Extrinsics

Uses pre-computed 2D animal keypoints as stereo correspondences to estimate
relative camera poses. No checkerboard required.

Strategy
--------
For each frame where the same animal is detected in 2+ cameras simultaneously
(hardware-synchronized), the matched body keypoints are correspondences of the
same 3D points. Collected across all configured sessions, then per camera pair:
  1. Undistort 2D points using known intrinsics
  2. findEssentialMat (RANSAC) to filter outliers
  3. recoverPose to get R, T

Camera 102 = reference (R=I, T=0). Others solved relative to 102.
Scale is relative (|T|=1) — intentional, sufficient for triangulation.

Sessions
--------
  250711  — Measurements/250711/2D_results  (cams 102, 108, 113, 117)
  250715  — BinaryData/250715               (cams 102, 108, 113, 117)
  250103  — BinaryData/250103               (cams 102→102, 10_→108, 113→113;
                                             101 has no match, dropped)

Output: DatalusCalibration/colmap_poses.npz

Usage:
    ./run.sh DatulusCalib_Step3_PoseCorrespondences.py
"""

import os
import sys
import re
import cv2
import numpy as np
from pathlib import Path
from itertools import combinations
from datetime import datetime

# ── Logging ───────────────────────────────────────────────────────────────────

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()
    def fileno(self): return self.streams[0].fileno()

def _setup_log(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "a")
    f.write(f"\n{'='*40}\n{datetime.now():%Y-%m-%d %H:%M:%S}  {Path(sys.argv[0]).name}\n{'='*40}\n")
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR       = "/Users/acalapai/PycharmProjects/Datalus"
INTRINSICS_NPZ = f"{BASE_DIR}/DatalusCalibration/intrinsics.npz"
OUTPUT_NPZ     = f"{BASE_DIR}/DatalusCalibration/colmap_poses.npz"

CAMERA_IDS     = ["102", "108", "113", "117"]
REFERENCE_CAM  = "102"

CONF_THRESH    = 0.5    # minimum keypoint confidence
MIN_PAIRS      = 50     # minimum correspondence pairs required per camera pair
RANSAC_THRESH  = 1e-3   # essential mat inlier threshold (normalized coords)

# Keypoint indices to use — avoid unstable extremities, prefer torso/head
# 0:nose 1:l_eye 2:r_eye 3:l_ear 4:r_ear 5:l_shldr 6:r_shldr
# 7:l_elbow 8:r_elbow 9:l_wrist 10:r_wrist 11:l_hip 12:r_hip
# 13:l_knee 14:r_knee 15:l_ankle 16:r_ankle
KEYPOINT_INDICES = list(range(17))   # use all; RANSAC handles outliers

# Force specific cameras to chain through a preferred anchor instead of the
# default BFS choice.  Key = camera to solve, value = required anchor.
# "108" is chained via "113" because the direct 102↔108 pair has only 6.5%
# RANSAC inliers, making its recovered pose unreliable.
CHAIN_OVERRIDES  = {"108": "113"}

# Sessions: each entry is (results_dir, cam_id_map)
# cam_id_map remaps file-level IDs to canonical CAMERA_IDS; identity entries can be omitted.
# Files with source IDs not in cam_id_map are skipped.
SESSIONS = [
    (
        f"{BASE_DIR}/Measurements/250711/2D_results",
        {"102": "102", "108": "108", "113": "113", "117": "117"},
    ),
    (
        f"{BASE_DIR}/BinaryData/250715",
        {"102": "102", "108": "108", "113": "113", "117": "117"},
    ),
    (
        f"{BASE_DIR}/BinaryData/250103",
        # 101 has no match in current cam set — dropped by omitting it
        {"102": "102", "10_": "108", "113": "113"},
    ),
]


# ── Parsing ───────────────────────────────────────────────────────────────────

def find_result_file(results_dir: str, src_cam_id: str):
    """Find a 2D result txt file for the given source camera ID in results_dir."""
    pattern = re.compile(rf".*__{src_cam_id}_.*_2D_result\.txt")
    matches = [p for p in Path(results_dir).glob("*.txt") if pattern.match(p.name)]
    if not matches:
        return None
    return matches[0]


def parse_results(path: Path) -> dict:
    """
    Returns dict: frame_num -> {monkey_id: kps (17, 3)}
    Data columns: frame monkey bbox×5 keypoints×(x,y,conf)×17
    """
    data = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            cols = line.split()
            if len(cols) < 7 + 17 * 3:
                continue
            frame  = int(cols[0])
            monkey = cols[1]
            kps    = np.array(cols[7:7 + 17*3], dtype=np.float32).reshape(17, 3)
            data.setdefault(frame, {})[monkey] = kps
    return data


def load_all_sessions(sessions: list) -> dict:
    """
    Load and merge 2D results from all sessions into a single per-camera dict.
    Each session uses an independent frame-number namespace to avoid collisions
    between sessions (offset by session_index * 10_000_000).

    Returns: {canonical_cam_id: {frame_key: {monkey_id: kps(17,3)}}}
    """
    merged = {cam: {} for cam in CAMERA_IDS}

    for sess_idx, (results_dir, cam_id_map) in enumerate(sessions):
        frame_offset = sess_idx * 10_000_000
        print(f"\n  Session [{sess_idx}] {Path(results_dir).name}")
        for src_id, canonical_id in cam_id_map.items():
            if canonical_id not in merged:
                continue
            path = find_result_file(results_dir, src_id)
            if path is None:
                print(f"    cam {src_id} → {canonical_id}: no file found, skipping")
                continue
            data = parse_results(path)
            n_frames = len(data)
            n_det    = sum(len(m) for m in data.values())
            # Apply frame offset so sessions don't share frame keys
            for frame, monkeys in data.items():
                merged[canonical_id][frame + frame_offset] = monkeys
            print(f"    cam {src_id} → {canonical_id}: {n_frames} frames, "
                  f"{n_det} detections  ← {path.name}")

    return merged


# ── Correspondence building ───────────────────────────────────────────────────

def build_correspondences(all_data: dict, cam_a: str, cam_b: str) -> tuple:
    """
    Find all frames where the same monkey is detected in both cam_a and cam_b.
    Returns (pts_a, pts_b) — arrays of shape (N, 2) of raw pixel coordinates.
    Only includes keypoints above CONF_THRESH in both cameras.
    """
    pts_a, pts_b = [], []
    data_a = all_data[cam_a]
    data_b = all_data[cam_b]

    shared_frames = set(data_a.keys()) & set(data_b.keys())
    for frame in shared_frames:
        shared_monkeys = set(data_a[frame].keys()) & set(data_b[frame].keys())
        for monkey in shared_monkeys:
            kps_a = data_a[frame][monkey]   # (17, 3)
            kps_b = data_b[frame][monkey]

            for k in KEYPOINT_INDICES:
                xa, ya, ca = kps_a[k]
                xb, yb, cb = kps_b[k]
                if ca >= CONF_THRESH and cb >= CONF_THRESH:
                    pts_a.append([xa, ya])
                    pts_b.append([xb, yb])

    return np.array(pts_a, dtype=np.float32), np.array(pts_b, dtype=np.float32)


# ── Essential matrix + pose recovery ─────────────────────────────────────────

def recover_relative_pose(pts_a: np.ndarray, pts_b: np.ndarray,
                           Ka: np.ndarray, da: np.ndarray,
                           Kb: np.ndarray, db: np.ndarray,
                           label: str):
    """
    Estimate R, T such that p_b = R @ p_a + T.
    Returns (R, T, n_inliers, inlier_ratio) or None on failure.
    """
    if len(pts_a) < MIN_PAIRS:
        print(f"  {label}: only {len(pts_a)} pairs — skipping (need {MIN_PAIRS})")
        return None

    # Undistort to normalised image coordinates (removes K and distortion)
    norm_a = cv2.undistortPoints(pts_a.reshape(-1,1,2), Ka, da).reshape(-1, 2)
    norm_b = cv2.undistortPoints(pts_b.reshape(-1,1,2), Kb, db).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(
        norm_a, norm_b,
        np.eye(3),                  # cameraMatrix (identity — already normalized)
        method=cv2.RANSAC, threshold=RANSAC_THRESH, prob=0.999
    )
    if E is None or mask is None:
        print(f"  {label}: findEssentialMat failed")
        return None

    n_inliers    = int(mask.sum())
    inlier_ratio = n_inliers / len(pts_a)

    _, R, T, mask2 = cv2.recoverPose(E, norm_a, norm_b, mask=mask)
    n_final = int(mask2.sum())

    print(f"  {label}: {len(pts_a)} pairs  →  "
          f"{n_inliers} inliers ({inlier_ratio*100:.1f}%)  →  "
          f"{n_final} recoverPose inliers")

    if n_inliers < 10:
        print(f"  {label}: too few inliers — skipping")
        return None

    return R, T.reshape(3,1), n_inliers, inlier_ratio


# ── Pose graph ────────────────────────────────────────────────────────────────

def build_pose_graph(all_data: dict, intrinsics: dict) -> dict:
    """
    Solve absolute world-to-camera poses for all cameras.
    Reference camera (REFERENCE_CAM) gets R=I, T=0.
    Others solved relative to reference via direct or chained estimation.
    Returns {cam_id: (R, T)}.
    """
    # Collect pairwise relative poses
    pairwise = {}
    print("\n[2] COMPUTING PAIRWISE RELATIVE POSES")
    for cam_a, cam_b in combinations(CAMERA_IDS, 2):
        print(f"\n  Pair {cam_a} ↔ {cam_b}:")
        pts_a, pts_b = build_correspondences(all_data, cam_a, cam_b)
        print(f"    {len(pts_a)} total correspondence points")
        result = recover_relative_pose(
            pts_a, pts_b,
            intrinsics[cam_a]["K"], intrinsics[cam_a]["dist"],
            intrinsics[cam_b]["K"], intrinsics[cam_b]["dist"],
            f"{cam_a}→{cam_b}"
        )
        if result is not None:
            R, T, n_in, ratio = result
            pairwise[(cam_a, cam_b)] = (R, T)
            pairwise[(cam_b, cam_a)] = (R.T, -R.T @ T)   # inverse

    # Build absolute poses starting from reference
    print(f"\n[3] BUILDING ABSOLUTE POSES (reference: cam {REFERENCE_CAM})")
    abs_poses = {REFERENCE_CAM: (np.eye(3), np.zeros((3,1)))}
    remaining = [c for c in CAMERA_IDS if c != REFERENCE_CAM]

    # BFS through pose graph, respecting CHAIN_OVERRIDES
    for _ in range(len(CAMERA_IDS)):
        if not remaining:
            break
        for cam in list(remaining):
            # If this camera has a forced anchor, wait until that anchor is solved
            forced_anchor = CHAIN_OVERRIDES.get(cam)
            if forced_anchor and forced_anchor not in abs_poses:
                continue   # anchor not solved yet — revisit in next BFS iteration
            anchors_to_try = (
                [forced_anchor] if forced_anchor
                else list(abs_poses.keys())
            )
            for anchor in anchors_to_try:
                if (anchor, cam) not in pairwise:
                    continue
                R_rel, T_rel = pairwise[(anchor, cam)]
                R_anc, T_anc = abs_poses[anchor]
                # p_cam = R_rel @ p_anchor + T_rel
                #       = R_rel @ (R_anc @ p_world + T_anc) + T_rel
                R_abs = R_rel @ R_anc
                T_abs = R_rel @ T_anc + T_rel
                abs_poses[cam] = (R_abs, T_abs)
                remaining.remove(cam)
                chain_note = f" (via override: {anchor})" if forced_anchor else ""
                print(f"  cam {anchor} → cam {cam}  ✓{chain_note}")
                break

    if remaining:
        print(f"  [WARN] Could not solve: {remaining}")

    return abs_poses


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(abs_poses: dict):
    print("\n[4] SANITY CHECK — Camera centres (world coords)")
    print(f"  {'CAM':>6}  {'Cx':>10}  {'Cy':>10}  {'Cz':>10}")
    centres = {}
    for cam_id, (R, T) in abs_poses.items():
        C = (-R.T @ T).ravel()
        centres[cam_id] = C
        print(f"  {cam_id:>6}  {C[0]:>10.4f}  {C[1]:>10.4f}  {C[2]:>10.4f}")

    print("\n  Pairwise distances (relative units, |T|=1 for direct pairs):")
    cams = list(centres.keys())
    for i in range(len(cams)):
        for j in range(i+1, len(cams)):
            ca, cb = cams[i], cams[j]
            d = np.linalg.norm(centres[ca] - centres[cb])
            print(f"    {ca} ↔ {cb}: {d:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _setup_log(str(Path(__file__).parent / "run.log"))
    print("=" * 60)
    print("DATALUS — STEP 3: POSE-CORRESPONDENCE EXTRINSICS")
    print("=" * 60)

    # Load intrinsics
    if not Path(INTRINSICS_NPZ).exists():
        print(f"[ERROR] {INTRINSICS_NPZ} not found. Run Step 2 first.")
        sys.exit(1)
    raw = np.load(INTRINSICS_NPZ)
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        if f"{cam_id}_K" not in raw:
            print(f"[ERROR] No intrinsics for cam {cam_id}")
            sys.exit(1)
        intrinsics[cam_id] = {
            "K":    raw[f"{cam_id}_K"],
            "dist": raw[f"{cam_id}_dist"].ravel(),
        }
    print("\n  Intrinsics loaded for:", list(intrinsics.keys()))

    # Parse 2D result files from all sessions
    print("\n[1] PARSING 2D RESULT FILES")
    all_data = load_all_sessions(SESSIONS)
    for cam_id in CAMERA_IDS:
        n_frames = len(all_data[cam_id])
        n_det    = sum(len(m) for m in all_data[cam_id].values())
        print(f"  cam {cam_id}: {n_frames} frames total, {n_det} detections (all sessions)")

    # Build pose graph
    abs_poses = build_pose_graph(all_data, intrinsics)

    if len(abs_poses) < 2:
        print("[ERROR] Could not solve poses for enough cameras.")
        sys.exit(1)

    # Sanity check
    sanity_check(abs_poses)

    # Save
    save_dict = {}
    for cam_id in CAMERA_IDS:
        if cam_id not in abs_poses:
            print(f"  [WARN] cam {cam_id} missing from solution — using identity")
            R, T = np.eye(3), np.zeros((3,1))
        else:
            R, T = abs_poses[cam_id]
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T.reshape(3,1)
        save_dict[f"{cam_id}_n"] = np.array([1])

    np.savez(OUTPUT_NPZ, **save_dict)
    print(f"\n  Poses saved → {OUTPUT_NPZ}")
    print("\nStep 3 complete. Run Step 4 next:")
    print("  ./run.sh DatulusCalib_Step4_WriteYAMLs.py")


if __name__ == "__main__":
    main()
