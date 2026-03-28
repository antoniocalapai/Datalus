#!/usr/bin/env python3
"""
Datalus Calibration — Step 3 (v2): Known-Position Extrinsics

Replaces the scale-ambiguous essential-matrix T vectors with metric translations
derived from physically measured camera positions (datalus_config.json).

Strategy
--------
  - Rotation matrices R: kept from the pose-correspondence step (colmap_poses.npz).
    These are geometrically consistent across cameras; only scale was wrong.
  - Translation vectors T: recomputed from measured room positions.
    Camera centre in world coords (mm, relative to cam 102):
        C = pos_cam - pos_102
    World-to-camera transform:
        T = -R @ C
    Convention: p_cam = R @ p_world + T

World origin = cam 102 physical position.
Units: millimetres.

Reads  : DatalusCalibration/colmap_poses.npz  (R matrices)
         DatalusCalibration/intrinsics.npz     (K, dist — for reprojection check)
         datalus_config.json                   (known positions)
Writes : DatalusCalibration/colmap_poses.npz  (updated T in mm)

Run Step 4 (DatulusCalib_Step4_WriteYAMLs.py) after this.
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR        = "/Users/acalapai/PycharmProjects/Datalus"
CONFIG_JSON     = f"{BASE_DIR}/datalus_config.json"
POSES_NPZ       = f"{BASE_DIR}/DatalusCalibration/colmap_poses_essmat.npz"   # R source
OUTPUT_POSES_NPZ = f"{BASE_DIR}/DatalusCalibration/colmap_poses.npz"          # output
INTRINSICS_NPZ  = f"{BASE_DIR}/DatalusCalibration/intrinsics.npz"
CAMERA_IDS     = ["102", "108", "113", "117"]
REFERENCE_CAM  = "102"


def main():
    print("=" * 60)
    print("DATALUS — STEP 3 (v2): KNOWN-POSITION EXTRINSICS")
    print("=" * 60)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = json.loads(Path(CONFIG_JSON).read_text())
    cam_positions_mm = {
        cam_id: np.array([
            cfg["cameras"][cam_id]["position_mm"]["x"],
            cfg["cameras"][cam_id]["position_mm"]["y"],
            cfg["cameras"][cam_id]["position_mm"]["z"],
        ], dtype=np.float64)
        for cam_id in CAMERA_IDS
    }
    ref_pos = cam_positions_mm[REFERENCE_CAM]
    print(f"\n  Reference camera: {REFERENCE_CAM} at {ref_pos} mm")
    print("\n  Camera world centres (relative to cam 102, mm):")
    world_centres = {}
    for cam_id in CAMERA_IDS:
        C = cam_positions_mm[cam_id] - ref_pos
        world_centres[cam_id] = C
        print(f"    cam {cam_id}: [{C[0]:>8.1f}, {C[1]:>8.1f}, {C[2]:>8.1f}]")

    # ── Load existing R matrices ───────────────────────────────────────────────
    if not Path(POSES_NPZ).exists():
        print(f"[ERROR] {POSES_NPZ} not found. Run Step 3 (pose correspondence) first.")
        sys.exit(1)
    poses = np.load(POSES_NPZ)
    print("\n  Rotation matrices loaded from pose-correspondence step.")

    # ── Compute new T = -R @ C ────────────────────────────────────────────────
    print("\n  Computing metric T vectors (mm):")
    save_dict = {}
    for cam_id in CAMERA_IDS:
        key_R = f"{cam_id}_R"
        if key_R not in poses:
            print(f"  [ERROR] No R for cam {cam_id} in {POSES_NPZ}")
            sys.exit(1)
        R = poses[key_R]
        C = world_centres[cam_id]
        T = -R @ C                    # T = -R C  →  p_cam = R p_world + T
        C_check = (-R.T @ T)          # should equal C

        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T.reshape(3, 1)
        save_dict[f"{cam_id}_n"] = poses.get(f"{cam_id}_n", np.array([0]))

        print(f"    cam {cam_id}: T = [{T[0]:>10.2f}, {T[1]:>10.2f}, {T[2]:>10.2f}] mm"
              f"   |C| = {np.linalg.norm(C):.1f} mm")

    # ── Sanity: pairwise distances ─────────────────────────────────────────────
    print("\n  Pairwise camera distances (mm):")
    for i, ca in enumerate(CAMERA_IDS):
        for cb in CAMERA_IDS[i+1:]:
            d = np.linalg.norm(world_centres[ca] - world_centres[cb])
            print(f"    {ca} ↔ {cb}: {d:.1f} mm  ({d/10:.1f} cm)")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez(OUTPUT_POSES_NPZ, **save_dict)
    print(f"\n  Saved → {OUTPUT_POSES_NPZ}")
    print("\nStep 3 (known positions) complete.")
    print("Run Step 4 next:  python3 DatulusCalib_Step4_WriteYAMLs.py")


if __name__ == "__main__":
    main()
