#!/usr/bin/env python3
"""
Datalus Calibration — Step 4: Write ABT-Compatible YAMLs

Reads intrinsics.npz (Step 2) and colmap_poses.npz (Step 3 Stereo) and
writes one OpenCV YAML per camera in the format ABT expects.

ABT convention (from ABT utils.py):
  - reads intrinsicMatrix row-major and TRANSPOSES → store K^T
  - reads R row-major and TRANSPOSES              → store R^T
  - T stored as-is (no transpose)

Poses are world-to-camera: p_cam = R @ p_world + T
Camera 108 is world origin (R=I, T=0).
Scale is in metres (square_size_mm / 1000 in Step 3).

Usage:
    ./run.sh DatulusCalib_Step4_WriteYAMLs.py
"""

import sys
import numpy as np
from pathlib import Path

INTRINSICS_NPZ = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/intrinsics.npz"
POSES_NPZ      = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/colmap_poses.npz"
OUTPUT_DIR     = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/yaml"
CAMERA_IDS     = ["102", "108", "113", "117"]


def fmt_matrix(arr: np.ndarray) -> str:
    vals = arr.ravel().tolist()
    return ", ".join(f"{v:.16e}" for v in vals)


def write_yaml(path: Path, K: np.ndarray, dist: np.ndarray,
               R: np.ndarray, T: np.ndarray):
    nd = dist.ravel().size
    path.write_text(
        f"%YAML:1.0\n"
        f"---\n"
        f"intrinsicMatrix: !!opencv-matrix\n"
        f"   rows: 3\n"
        f"   cols: 3\n"
        f"   dt: d\n"
        f"   data: [ {fmt_matrix(K.T)} ]\n"
        f"distortionCoefficients: !!opencv-matrix\n"
        f"   rows: 1\n"
        f"   cols: {nd}\n"
        f"   dt: d\n"
        f"   data: [ {fmt_matrix(dist.ravel())} ]\n"
        f"R: !!opencv-matrix\n"
        f"   rows: 3\n"
        f"   cols: 3\n"
        f"   dt: d\n"
        f"   data: [ {fmt_matrix(R.T)} ]\n"
        f"T: !!opencv-matrix\n"
        f"   rows: 3\n"
        f"   cols: 1\n"
        f"   dt: d\n"
        f"   data: [ {fmt_matrix(T.ravel())} ]\n"
    )


def main():
    for p, label in [(INTRINSICS_NPZ, "intrinsics.npz"),
                     (POSES_NPZ,      "colmap_poses.npz")]:
        if not Path(p).exists():
            print(f"[ERROR] {label} not found: {p}")
            print("  Run Step 2 then Step 3 (Stereo) first.")
            sys.exit(1)

    intr  = np.load(INTRINSICS_NPZ)
    poses = np.load(POSES_NPZ)

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    centres = {}
    print(f"\n{'CAM':>6}  {'RMS (px)':>10}  {'Cx (m)':>10}  {'Cy (m)':>10}  {'Cz (m)':>10}")
    print("-" * 56)

    for cam_id in CAMERA_IDS:
        if f"{cam_id}_K" not in intr:
            print(f"  [WARN] No intrinsics for cam {cam_id} — skipping")
            continue
        if f"{cam_id}_R" not in poses:
            print(f"  [WARN] No pose for cam {cam_id} — skipping")
            continue

        K    = intr[f"{cam_id}_K"]
        dist = intr[f"{cam_id}_dist"].ravel()
        rms  = float(intr[f"{cam_id}_rms"].ravel()[0])
        R    = poses[f"{cam_id}_R"]
        T    = poses[f"{cam_id}_T"].reshape(3, 1)

        # Camera centre in world coords: C = -R^T @ T
        C = (-R.T @ T).ravel()
        centres[cam_id] = C

        yaml_path = out / f"{cam_id}.yaml"
        write_yaml(yaml_path, K, dist, R, T)

        print(f"  {cam_id:>4}  {rms:>10.2f}  {C[0]:>10.4f}  {C[1]:>10.4f}  {C[2]:>10.4f}  → {yaml_path.name}")

    print()
    print("Pairwise camera separations:")
    cams = list(centres.keys())
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            ca, cb = cams[i], cams[j]
            d = np.linalg.norm(centres[ca] - centres[cb])
            print(f"  cam {ca} <-> cam {cb}: {d:.3f} m  ({d * 100:.1f} cm)")

    print(f"\nYAMLs written to {out}/")
    print("Next: run Datalus3D_Visualize.py with 4 synchronized images or videos.")


if __name__ == "__main__":
    main()
