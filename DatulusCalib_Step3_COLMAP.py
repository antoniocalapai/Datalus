#!/usr/bin/env python3
"""
Datalus Calibration — Step 3: COLMAP Sparse Reconstruction
Runs COLMAP on extracted frames with fixed per-camera intrinsics from Step 2.
Outputs per-camera R and T in a consistent coordinate system.
Run after DatulusCalib_Step2_Intrinsics.py.
"""

import os
import sys
import shutil
import subprocess
import numpy as np
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

FRAMES_ROOT        = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration/frames"
INTRINSICS_NPZ     = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration/intrinsics.npz"
OUTPUT_DIR         = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"
CAMERA_IDS         = ["102", "108", "113", "117"]

MAX_FRAMES_PER_CAM = 150   # evenly subsample to this many frames per camera
N_THREADS          = 12    # 75% of 16 cores

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def find_colmap():
    for candidate in ["/opt/homebrew/bin/colmap", "/usr/local/bin/colmap"]:
        if os.path.isfile(candidate):
            return candidate
    found = shutil.which("colmap")
    if found:
        return found
    print("[ERROR] colmap not found. Install with: brew install colmap")
    sys.exit(1)


def run(cmd, desc):
    print(f"\n  >> {desc}")
    print(f"     {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd], capture_output=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed (exit {result.returncode}): {desc}")
        sys.exit(1)


def load_intrinsics(npz_path):
    """Load intrinsics saved by Step 2. Returns dict cam_id -> (K, dist)."""
    data = np.load(npz_path)
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        k = f"{cam_id}_K"
        d = f"{cam_id}_dist"
        if k in data and d in data:
            intrinsics[cam_id] = (data[k], data[d])
        else:
            print(f"  [WARN] No intrinsics found for cam {cam_id} in {npz_path}")
    return intrinsics


def build_image_lists(frames_root, camera_ids, max_per_cam, colmap_dir):
    """
    For each camera, build a text file listing relative image paths (relative
    to frames_root). Subsample to max_per_cam evenly spaced frames.
    Returns dict cam_id -> list_file_path.
    """
    lists = {}
    for cam_id in camera_ids:
        cam_dir = Path(frames_root) / cam_id
        all_frames = sorted(cam_dir.glob("*.png"))
        if not all_frames:
            print(f"  [WARN] No frames for cam {cam_id}")
            continue

        # evenly subsample
        n = len(all_frames)
        if n > max_per_cam:
            indices = np.linspace(0, n - 1, max_per_cam, dtype=int)
            selected = [all_frames[i] for i in indices]
        else:
            selected = all_frames

        list_path = colmap_dir / f"list_{cam_id}.txt"
        with open(list_path, "w") as f:
            for img in selected:
                # path relative to frames_root
                f.write(str(img.relative_to(frames_root)) + "\n")

        lists[cam_id] = list_path
        print(f"  cam {cam_id}: {len(selected)} / {n} frames selected")

    return lists


def quat_to_rot(qw, qx, qy, qz):
    """Unit quaternion (COLMAP convention) to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ])


def parse_colmap_images(images_txt):
    """
    Parse COLMAP images.txt.
    Returns dict: image_name -> (R 3x3, T 3x1)  world-to-camera.
    """
    poses = {}
    with open(images_txt) as f:
        lines = [l.strip() for l in f if not l.startswith("#") and l.strip()]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz     = float(parts[5]), float(parts[6]), float(parts[7])
        name           = parts[9]
        R = quat_to_rot(qw, qx, qy, qz)
        T = np.array([[tx], [ty], [tz]])
        poses[name] = (R, T)
        i += 2  # skip the keypoints line
    return poses


def average_rotations(R_list):
    """Geodesic mean of rotation matrices via SVD."""
    R_sum = sum(R_list)
    U, _, Vt = np.linalg.svd(R_sum)
    return U @ Vt


def aggregate_poses(poses, camera_ids):
    """Average R and T across all frames of the same camera."""
    cam_poses = {}
    for cam_id in camera_ids:
        R_list, T_list = [], []
        for name, (R, T) in poses.items():
            folder = Path(name).parts[0]
            if folder == cam_id:
                R_list.append(R)
                T_list.append(T)

        if not R_list:
            print(f"  [WARN] No reconstructed frames for cam {cam_id}")
            continue

        R_mean = average_rotations(R_list)
        T_mean = np.mean(T_list, axis=0)
        cam_poses[cam_id] = (R_mean, T_mean)
        print(f"  cam {cam_id}: {len(R_list)} frames averaged into final pose")

    return cam_poses


def best_model(sparse_dir):
    """Return the model subdirectory with the most registered images."""
    models = sorted(sparse_dir.iterdir())
    if not models:
        print("[ERROR] COLMAP produced no models. Check feature matches.")
        sys.exit(1)
    if len(models) == 1:
        return models[0]
    # pick the one whose images.txt has the most entries
    best, best_n = models[0], 0
    for m in models:
        txt = m / "images.txt"
        if txt.exists():
            with open(txt) as f:
                n = sum(1 for l in f if not l.startswith("#") and l.strip()) // 2
            if n > best_n:
                best, best_n = m, n
    print(f"  Selected model: {best.name} ({best_n} registered images)")
    return best


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DATALUS — STEP 3: COLMAP SPARSE RECONSTRUCTION")
    print("=" * 60)

    colmap = find_colmap()
    print(f"\n  COLMAP binary: {colmap}")

    # ── Directories ──
    colmap_dir = Path(OUTPUT_DIR) / "colmap"
    sparse_dir = colmap_dir / "sparse"
    txt_dir    = colmap_dir / "sparse_txt"
    db_path    = colmap_dir / "database.db"

    colmap_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)
    txt_dir.mkdir(exist_ok=True)

    # ── Load intrinsics ──
    print("\n  Loading intrinsics from Step 2...")
    if not os.path.exists(INTRINSICS_NPZ):
        print(f"[ERROR] {INTRINSICS_NPZ} not found. Run Step 2 first.")
        sys.exit(1)
    intrinsics = load_intrinsics(INTRINSICS_NPZ)

    # ── Build per-camera image lists ──
    print(f"\n  Building image lists (max {MAX_FRAMES_PER_CAM} frames/camera)...")
    image_lists = build_image_lists(FRAMES_ROOT, CAMERA_IDS, MAX_FRAMES_PER_CAM, colmap_dir)

    # ── Create database ──
    if db_path.exists():
        db_path.unlink()
    run([colmap, "database_creator", "--database_path", db_path],
        "Creating COLMAP database")

    # ── Feature extraction per camera with fixed intrinsics ──
    print("\n  Extracting features (one camera at a time with fixed intrinsics)...")
    for cam_id in CAMERA_IDS:
        if cam_id not in intrinsics or cam_id not in image_lists:
            print(f"  [SKIP] cam {cam_id}: missing intrinsics or frames")
            continue

        K, dist = intrinsics[cam_id]
        fx, fy   = K[0, 0], K[1, 1]
        cx, cy   = K[0, 2], K[1, 2]
        k1, k2, p1, p2 = dist.ravel()[:4]
        params = f"{fx},{fy},{cx},{cy},{k1},{k2},{p1},{p2}"

        run([
            colmap, "feature_extractor",
            "--database_path",                        db_path,
            "--image_path",                           FRAMES_ROOT,
            "--image_list_path",                      image_lists[cam_id],
            "--ImageReader.camera_model",             "OPENCV",
            "--ImageReader.camera_params",            params,
            "--ImageReader.single_camera_per_folder", "1",

            "--FeatureExtraction.use_gpu",             "0",
            "--FeatureExtraction.num_threads",        N_THREADS,
        ], f"Feature extraction: cam {cam_id}")

    # ── Exhaustive matching ──
    run([
        colmap, "exhaustive_matcher",
        "--database_path",        db_path,
        "--FeatureMatching.use_gpu", "0",
        "--FeatureMatching.num_threads", N_THREADS,
    ], "Exhaustive feature matching")

    # ── Sparse reconstruction ──
    run([
        colmap, "mapper",
        "--database_path",                    db_path,
        "--image_path",                       FRAMES_ROOT,
        "--output_path",                      sparse_dir,
        "--Mapper.ba_refine_focal_length",    "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params",    "0",
    ], "Sparse reconstruction (mapper)")

    # ── Convert best model to TXT ──
    model = best_model(sparse_dir)
    run([
        colmap, "model_converter",
        "--input_path",  model,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ], "Converting model to TXT")

    # ── Parse and aggregate poses ──
    print("\n  Parsing camera poses...")
    images_txt = txt_dir / "images.txt"
    if not images_txt.exists():
        print(f"[ERROR] {images_txt} not found after conversion.")
        sys.exit(1)

    all_poses  = parse_colmap_images(images_txt)
    print(f"  Total reconstructed images: {len(all_poses)}")
    cam_poses  = aggregate_poses(all_poses, CAMERA_IDS)

    if not cam_poses:
        print("[ERROR] No camera poses recovered. COLMAP may not have registered all cameras.")
        sys.exit(1)

    # ── Save poses ──
    out_path = Path(OUTPUT_DIR) / "colmap_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in cam_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
    np.savez(str(out_path), **save_dict)
    print(f"\n  Poses saved to {out_path}")

    print("\nSummary:")
    for cam_id, (R, T) in cam_poses.items():
        C = -R.T @ T  # camera centre in world coords
        print(f"  cam {cam_id}: centre = [{C[0,0]:.2f}, {C[1,0]:.2f}, {C[2,0]:.2f}]")

    print("\nStep 3 complete.")


if __name__ == "__main__":
    main()