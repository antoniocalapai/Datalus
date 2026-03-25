#!/usr/bin/env python3
"""
Datalus Calibration — Step 4: World Registration
Fits a similarity transform (scale + rotation + translation) from COLMAP space
to real-world mm coordinates using manually identified reference points.
Applies the transform to all camera poses and saves world_poses.npz.

Run after DatulusCalib_Step3_COLMAP.py.

Workflow:
  1. Run Step 3 to produce colmap_poses.npz and colmap/sparse_txt/points3D.txt
  2. Open colmap/sparse_txt/points3D.txt — find 3+ stable, identifiable points
     (e.g. cage corners, fiducial markers) and note their POINT3D_ID and XYZ
  3. Measure those same points in real-world mm (with a ruler or known geometry)
  4. Fill in world_registration.csv (created as a template on first run):
       name, colmap_x, colmap_y, colmap_z, real_x_mm, real_y_mm, real_z_mm
  5. Re-run this script
"""

import os
import sys
import csv
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

OUTPUT_DIR       = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"
COLMAP_POSES_NPZ = os.path.join(OUTPUT_DIR, "colmap_poses.npz")
WORLD_CSV        = os.path.join(OUTPUT_DIR, "world_registration.csv")
POINTS3D_TXT     = os.path.join(OUTPUT_DIR, "colmap", "sparse_txt", "points3D.txt")

CAMERA_IDS       = ["102", "108", "113", "117"]

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def umeyama(src, dst):
    """
    Umeyama (1991) similarity transform: dst ≈ s * R @ src + t
    src, dst : (N, 3) float64 arrays  (N >= 3)
    Returns  : s (float), R (3x3 ndarray), t (3,) ndarray
    """
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)

    src_c = src - mu_s
    dst_c = dst - mu_d

    sigma_s2 = (src_c ** 2).sum() / n
    Sigma    = (dst_c.T @ src_c) / n

    U, d, Vt = np.linalg.svd(Sigma)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = np.trace(np.diag(d) @ S) / sigma_s2
    t = mu_d - s * R @ mu_s

    return s, R, t


def transform_pose(R_c, T_c, s, R_w, t_w):
    """
    Convert a COLMAP camera pose (R_c, T_c) to world-mm pose (R_out, T_out).

    COLMAP convention:  p_cam = R_c @ p_colmap + T_c
    World transform:    p_world_mm = s * R_w @ p_colmap + t_w

    Resulting world-to-cam extrinsic (p_cam in mm-scale units):
        R_out = R_c @ R_w.T
        T_out = s * T_c - R_out @ t_w
    """
    R_out = R_c @ R_w.T
    T_out = s * T_c - R_out @ t_w.reshape(3, 1)
    return R_out, T_out


def load_colmap_poses(npz_path, camera_ids):
    data = np.load(npz_path)
    poses = {}
    for cam_id in camera_ids:
        kr, kt = f"{cam_id}_R", f"{cam_id}_T"
        if kr in data and kt in data:
            poses[cam_id] = (data[kr], data[kt])
        else:
            print(f"  [WARN] No COLMAP pose for cam {cam_id}")
    return poses


def load_world_csv(csv_path):
    """
    Returns src (N,3) COLMAP XYZ and dst (N,3) real-world mm, plus names list.
    """
    src, dst, names = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # skip template/comment rows
            if row["colmap_x"].strip().startswith("#") or row["colmap_x"].strip() == "":
                continue
            names.append(row["name"].strip())
            src.append([float(row["colmap_x"]), float(row["colmap_y"]), float(row["colmap_z"])])
            dst.append([float(row["real_x_mm"]), float(row["real_y_mm"]), float(row["real_z_mm"])])
    return np.array(src, dtype=np.float64), np.array(dst, dtype=np.float64), names


def print_points3d_sample(path, n=30):
    """Print the first n 3D points from COLMAP points3D.txt to help identify reference pts."""
    if not os.path.exists(path):
        print(f"  [INFO] points3D.txt not found at {path}")
        return
    print(f"\n  First {n} COLMAP 3D points (use these to fill world_registration.csv):")
    print(f"  {'POINT3D_ID':>12}  {'X':>10}  {'Y':>10}  {'Z':>10}  N_TRACKS")
    count = 0
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pid = parts[0]
            x, y, z = parts[1], parts[2], parts[3]
            n_tracks = len(parts[8:]) // 2 if len(parts) > 8 else 0
            print(f"  {pid:>12}  {float(x):>10.4f}  {float(y):>10.4f}  {float(z):>10.4f}  {n_tracks}")
            count += 1
            if count >= n:
                break


def create_csv_template(csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "colmap_x", "colmap_y", "colmap_z",
                         "real_x_mm", "real_y_mm", "real_z_mm"])
        writer.writerow(["point_A", "", "", "", "", "", ""])
        writer.writerow(["point_B", "", "", "", "", "", ""])
        writer.writerow(["point_C", "", "", "", "", "", ""])
    print(f"\n  Template created: {csv_path}")
    print("  Fill in colmap_x/y/z from points3D.txt (printed above).")
    print("  Fill in real_x/y/z in mm from physical measurements.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DATALUS — STEP 4: WORLD REGISTRATION")
    print("=" * 60)

    # ── Check COLMAP poses exist ──
    if not os.path.exists(COLMAP_POSES_NPZ):
        print(f"[ERROR] {COLMAP_POSES_NPZ} not found. Run Step 3 first.")
        sys.exit(1)

    # ── If CSV missing or empty, create template and show COLMAP points ──
    csv_ready = False
    if os.path.exists(WORLD_CSV):
        try:
            src, dst, names = load_world_csv(WORLD_CSV)
            if len(names) >= 3:
                csv_ready = True
        except Exception:
            pass

    if not csv_ready:
        print("\n  world_registration.csv not ready (missing, empty, or < 3 points).")
        print_points3d_sample(POINTS3D_TXT)
        create_csv_template(WORLD_CSV)
        print("\n  Fill in the CSV and re-run this script.")
        sys.exit(0)

    src, dst, names = load_world_csv(WORLD_CSV)
    print(f"\n  Loaded {len(names)} reference points: {names}")

    # ── Fit similarity transform ──
    s, R_w, t_w = umeyama(src, dst)
    print(f"\n  Umeyama fit:")
    print(f"    scale  = {s:.6f}  ({s:.4f} mm per COLMAP unit)")
    print(f"    R_w    =")
    for row in R_w:
        print(f"             [{row[0]:10.6f}  {row[1]:10.6f}  {row[2]:10.6f}]")
    print(f"    t_w    = [{t_w[0]:.3f}, {t_w[1]:.3f}, {t_w[2]:.3f}] mm")

    # ── Registration residuals ──
    residuals = []
    print(f"\n  Registration residuals (mm):")
    for i, name in enumerate(names):
        p_est = s * R_w @ src[i] + t_w
        err   = np.linalg.norm(p_est - dst[i])
        residuals.append(err)
        print(f"    {name:20s}  est=[{p_est[0]:.2f},{p_est[1]:.2f},{p_est[2]:.2f}]"
              f"  gt=[{dst[i,0]:.2f},{dst[i,1]:.2f},{dst[i,2]:.2f}]  err={err:.3f} mm")
    print(f"    RMSE = {np.sqrt(np.mean(np.array(residuals)**2)):.3f} mm")

    # ── Load and transform camera poses ──
    colmap_poses = load_colmap_poses(COLMAP_POSES_NPZ, CAMERA_IDS)
    world_poses  = {}

    print(f"\n  Camera centres in world (mm):")
    for cam_id, (R_c, T_c) in colmap_poses.items():
        R_out, T_out = transform_pose(R_c, T_c, s, R_w, t_w)
        world_poses[cam_id] = (R_out, T_out)
        # camera centre: C = -R.T @ T
        C = -R_out.T @ T_out
        print(f"    cam {cam_id}: [{C[0,0]:.2f}, {C[1,0]:.2f}, {C[2,0]:.2f}] mm")

    # ── Save ──
    out_path = Path(OUTPUT_DIR) / "world_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in world_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
    np.savez(str(out_path), **save_dict)
    print(f"\n  World poses saved to {out_path}")

    print("\nStep 4 complete.")


if __name__ == "__main__":
    main()