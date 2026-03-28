#!/usr/bin/env python3
"""
Datalus Calibration — Step 3 (Room-Based): Checkerboard-Free Camera Pose Estimation

Strategy
--------
Instead of requiring a checkerboard to be simultaneously visible in all cameras,
we use a phone video walking through the room as a "bridge":

  1. Extract ~150 frames from the phone video (full room coverage)
  2. Sample ~15 frames from each fixed camera (room texture, any footage)
  3. Feed all ~210 images into COLMAP with:
       - Fixed cameras: OPENCV model, intrinsics locked from Step 2
       - Phone: SIMPLE_RADIAL, intrinsics free
  4. COLMAP connects all cameras through shared room features
  5. Extract and average fixed-camera poses from the reconstruction

Output: colmap_poses.npz — same format as before, compatible with Step 4.

Usage
-----
1. Drop your phone walkthrough video into PHONE_VIDEO_DIR
2. Run:  ./run.sh DatulusCalib_Step3_RoomBased.py

Run after DatulusCalib_Step2_Intrinsics.py.
"""

import os
import re
import sys
import shutil
import subprocess
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
    _f.write(f"\n{'='*40}\n{datetime.now():%Y-%m-%d %H:%M:%S}  {Path(sys.argv[0]).name}\n{'='*40}\n")
    sys.stdout = _Tee(sys.__stdout__, _f)
    sys.stderr = _Tee(sys.__stderr__, _f)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

PHONE_VIDEO_DIR      = "/Users/acalapai/PycharmProjects/Datalus"
INTRINSICS_NPZ       = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/intrinsics.npz"
OUTPUT_DIR           = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration"

CAMERA_IDS           = ["102", "108", "113", "117"]

# Empty-room videos: one per camera, named with cam ID in filename
ROOM_VIDEO_DIR       = "/Users/acalapai/PycharmProjects/Datalus/Measurements/250711/RAW"
ROOM_VIDEO_PATTERN   = "July_2025__{cam_id}_20250711151000.mp4"

# Extract frames from this time window (room is empty at minute 15 for ~15 min)
ROOM_EMPTY_START_SEC = 15 * 60   # 900 s
ROOM_EMPTY_END_SEC   = 25 * 60   # 1500 s
ROOM_FRAMES_PER_CAM  = 5         # frames to extract per fixed camera

PHONE_FRAMES         = 1000   # frames to extract from phone video
N_THREADS            = 12

PNG_PARAMS           = [cv2.IMWRITE_PNG_COMPRESSION, 0]
VIDEO_EXTENSIONS     = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def find_colmap() -> str:
    for c in ["/opt/homebrew/bin/colmap", "/usr/local/bin/colmap"]:
        if os.path.isfile(c):
            return c
    found = shutil.which("colmap")
    if found:
        return found
    print("[ERROR] colmap not found.  brew install colmap")
    sys.exit(1)


def run(cmd, desc: str):
    print(f"\n  >> {desc}")
    print(f"     {' '.join(str(c) for c in cmd)}")
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        print(f"\n[ERROR] {desc} failed (exit {r.returncode})")
        sys.exit(1)


def find_video(video_dir: str) -> Path:
    videos = sorted(
        p for p in Path(video_dir).iterdir()
        if p.suffix in VIDEO_EXTENSIONS and not p.name.startswith(".")
    )
    if not videos:
        print(f"[ERROR] No video found in {video_dir}")
        print(f"  Drop a .mp4 / .mov of you walking through the room there and re-run.")
        sys.exit(1)
    if len(videos) == 1:
        return videos[0]
    print("\n  Multiple videos found:")
    for i, v in enumerate(videos):
        print(f"    [{i}] {v.name}")
    return videos[int(input("  Select index: "))]


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
        [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
        [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)],
    ])


def average_rotations(R_list: list) -> np.ndarray:
    U, _, Vt = np.linalg.svd(sum(R_list))
    return U @ Vt


# ─── FRAME SAMPLING ───────────────────────────────────────────────────────────

def extract_phone_frames(video_path: Path, out_dir: Path, n_frames: int):
    """Extract n_frames evenly spaced frames from the phone video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    indices = set(np.linspace(0, total - 1, n_frames, dtype=int))
    saved   = 0

    print(f"  Phone: {video_path.name}  {w}×{h}  {fps:.0f}fps  "
          f"{total} frames  →  sampling {n_frames}")

    for idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.png"), frame, PNG_PARAMS)
            saved += 1
        if idx % 500 == 0:
            sys.stdout.write(f"\r  Extracting phone frames... {saved}/{n_frames}")
            sys.stdout.flush()

    cap.release()
    print(f"\r  Extracted {saved} phone frames                    ")
    return saved, (w, h)


def extract_room_frames(cam_id: str, out_dir: Path) -> int:
    """Extract ROOM_FRAMES_PER_CAM evenly-spaced frames from the empty-room
    window [ROOM_EMPTY_START_SEC, ROOM_EMPTY_END_SEC] of the fixed camera video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(ROOM_VIDEO_DIR) / ROOM_VIDEO_PATTERN.format(cam_id=cam_id)
    if not video_path.exists():
        print(f"  [WARN] Room video not found for cam {cam_id}: {video_path}")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 5.0

    start_frame = int(ROOM_EMPTY_START_SEC * fps)
    end_frame   = int(ROOM_EMPTY_END_SEC   * fps)
    indices     = set(np.linspace(start_frame, end_frame, ROOM_FRAMES_PER_CAM, dtype=int))

    saved = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            cv2.imwrite(str(out_dir / f"room_{idx:06d}.png"), frame, PNG_PARAMS)
            saved += 1
        if saved == ROOM_FRAMES_PER_CAM:
            break

    cap.release()
    print(f"  cam {cam_id}: extracted {saved} room frames "
          f"(t={ROOM_EMPTY_START_SEC//60}-{ROOM_EMPTY_END_SEC//60} min)")
    return saved


# ─── COLMAP ───────────────────────────────────────────────────────────────────

def feature_extract_group(colmap, db, image_root, list_path,
                           camera_model, camera_params=None):
    """Run feature extraction for one group of images (one camera model)."""
    cmd = [
        colmap, "feature_extractor",
        "--database_path",                 db,
        "--image_path",                    image_root,
        "--image_list_path",               list_path,
        "--ImageReader.camera_model",      camera_model,
        "--ImageReader.single_camera",     "1",
        "--FeatureExtraction.use_gpu",     "0",
        "--FeatureExtraction.num_threads", N_THREADS,
    ]
    if camera_params:
        cmd += ["--ImageReader.camera_params", camera_params]
    run(cmd, f"Feature extraction: {list_path.stem}")


def parse_images_txt(images_txt: Path) -> dict:
    """Parse COLMAP images.txt → {image_name: (R 3×3, T 3×1)}."""
    poses = {}
    lines = [l.strip() for l in open(images_txt)
             if not l.startswith("#") and l.strip()]
    i = 0
    while i < len(lines):
        p  = lines[i].split()
        qw, qx, qy, qz = map(float, p[1:5])
        tx, ty, tz      = map(float, p[5:8])
        name            = p[9]
        poses[name]     = (quat_to_rot(qw, qx, qy, qz),
                           np.array([[tx], [ty], [tz]]))
        i += 2
    return poses


def best_model(sparse_dir: Path) -> Path:
    import struct
    models = sorted(sparse_dir.iterdir())
    if not models:
        print("[ERROR] COLMAP produced no models.")
        sys.exit(1)
    def count(m):
        b = m / "images.bin"
        if not b.exists():
            return 0
        with open(b, "rb") as f:
            return struct.unpack("<Q", f.read(8))[0]
    best = max(models, key=count)
    print(f"  Best model: {best.name}  ({count(best)} registered images)")
    return best


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _setup_log(str(Path(__file__).parent / "run.log"))
    print("=" * 60)
    print("DATALUS — STEP 3: ROOM-BASED CAMERA POSE ESTIMATION")
    print("=" * 60)

    colmap = find_colmap()
    print(f"\n  COLMAP: {colmap}")

    # ── Load intrinsics ────────────────────────────────────────────────────────
    if not os.path.exists(INTRINSICS_NPZ):
        print(f"[ERROR] {INTRINSICS_NPZ} not found. Run Step 2 first.")
        sys.exit(1)
    raw        = np.load(INTRINSICS_NPZ)
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        if f"{cam_id}_K" in raw:
            K    = raw[f"{cam_id}_K"]
            dist = raw[f"{cam_id}_dist"].ravel()
            intrinsics[cam_id] = {
                "K": K, "dist": dist,
                "params": f"{K[0,0]},{K[1,1]},{K[0,2]},{K[1,2]},"
                          f"{dist[0]},{dist[1]},{dist[2]},{dist[3]}",
            }
        else:
            print(f"  [WARN] No intrinsics for cam {cam_id}")

    # ── Directories ────────────────────────────────────────────────────────────
    colmap_dir = Path(OUTPUT_DIR) / "colmap_room"
    images_dir = colmap_dir / "images"
    sparse_dir = colmap_dir / "sparse"
    txt_dir    = colmap_dir / "sparse_txt"
    db_path    = colmap_dir / "database.db"
    lists_dir  = colmap_dir / "lists"

    # Always start fresh to avoid stale data from previous runs
    if colmap_dir.exists():
        shutil.rmtree(str(colmap_dir))

    for d in [images_dir, sparse_dir, txt_dir, lists_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Find phone video ───────────────────────────────────────────────────────
    print(f"\n  Scanning {PHONE_VIDEO_DIR} for phone video...")
    video_path = find_video(PHONE_VIDEO_DIR)
    print(f"  Using: {video_path.name}")

    # ── Sample frames ──────────────────────────────────────────────────────────
    print("\n[1] SAMPLING FRAMES")

    phone_dir = images_dir / "phone"
    extract_phone_frames(video_path, phone_dir, PHONE_FRAMES)

    for cam_id in CAMERA_IDS:
        extract_room_frames(cam_id, images_dir / cam_id)

    # ── Build image lists ──────────────────────────────────────────────────────
    # Paths are relative to images_dir (COLMAP image root)
    phone_list = lists_dir / "list_phone.txt"
    with open(phone_list, "w") as f:
        for p in sorted(phone_dir.glob("*.png")):
            f.write(f"phone/{p.name}\n")

    cam_lists = {}
    for cam_id in CAMERA_IDS:
        list_path = lists_dir / f"list_{cam_id}.txt"
        with open(list_path, "w") as f:
            for p in sorted((images_dir / cam_id).glob("*.png")):
                f.write(f"{cam_id}/{p.name}\n")
        cam_lists[cam_id] = list_path

    # ── Feature extraction ─────────────────────────────────────────────────────
    print("\n[2] FEATURE EXTRACTION")

    if db_path.exists():
        db_path.unlink()

    # Phone — free intrinsics (SIMPLE_RADIAL)
    feature_extract_group(colmap, db_path, images_dir, phone_list,
                          "SIMPLE_RADIAL")

    # Fixed cameras — locked intrinsics (OPENCV)
    for cam_id in CAMERA_IDS:
        if cam_id not in intrinsics:
            continue
        feature_extract_group(colmap, db_path, images_dir,
                              cam_lists[cam_id], "OPENCV",
                              intrinsics[cam_id]["params"])

    # ── Matching ───────────────────────────────────────────────────────────────
    print("\n[3] FEATURE MATCHING")

    # Pass 1: sequential matching on phone frames — builds a solid video chain
    run([
        colmap, "sequential_matcher",
        "--database_path",                     db_path,
        "--SequentialMatching.overlap",        "15",
        "--SequentialMatching.loop_detection", "1",
        "--FeatureMatching.use_gpu",           "0",
        "--FeatureMatching.num_threads",       N_THREADS,
    ], "Sequential matching (phone frames)")

    # Pass 2: exhaustive matching across all images — connects fixed cams to phone
    run([
        colmap, "exhaustive_matcher",
        "--database_path",               db_path,
        "--FeatureMatching.use_gpu",     "0",
        "--FeatureMatching.num_threads", N_THREADS,
    ], "Exhaustive matching (all cameras)")

    # ── Sparse reconstruction ──────────────────────────────────────────────────
    print("\n[4] SPARSE RECONSTRUCTION")
    run([
        colmap, "mapper",
        "--database_path",                    db_path,
        "--image_path",                       images_dir,
        "--output_path",                      sparse_dir,
        "--Mapper.ba_refine_focal_length",    "1",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params",    "1",
    ], "Sparse reconstruction (mapper)")

    model = best_model(sparse_dir)

    run([
        colmap, "model_converter",
        "--input_path",  model,
        "--output_path", txt_dir,
        "--output_type", "TXT",
    ], "Convert to TXT")

    # ── Extract fixed camera poses ─────────────────────────────────────────────
    print("\n[5] EXTRACTING CAMERA POSES")
    images_txt = txt_dir / "images.txt"
    if not images_txt.exists():
        print(f"[ERROR] {images_txt} not found.")
        sys.exit(1)

    all_poses = parse_images_txt(images_txt)
    print(f"  Total registered images: {len(all_poses)}")

    cam_poses    = {}
    cam_n_frames = {}

    for cam_id in CAMERA_IDS:
        R_list, T_list = [], []
        for name, (R, T) in all_poses.items():
            if Path(name).parts[0] == cam_id:
                R_list.append(R)
                T_list.append(T)
        if not R_list:
            print(f"  [WARN] No registered frames for cam {cam_id}")
            continue
        cam_poses[cam_id]    = (average_rotations(R_list),
                                np.mean(T_list, axis=0))
        cam_n_frames[cam_id] = len(R_list)
        print(f"  cam {cam_id}: {len(R_list)} frames registered → averaged pose")

    if not cam_poses:
        print("[ERROR] No camera poses recovered. "
              "Check that the phone video covers the full room.")
        sys.exit(1)

    # ── Save ───────────────────────────────────────────────────────────────────
    out_path  = Path(OUTPUT_DIR) / "colmap_poses.npz"
    save_dict = {}
    for cam_id, (R, T) in cam_poses.items():
        save_dict[f"{cam_id}_R"] = R
        save_dict[f"{cam_id}_T"] = T
        save_dict[f"{cam_id}_n"] = np.array([cam_n_frames[cam_id]])
    np.savez(str(out_path), **save_dict)
    print(f"\n  Poses saved -> {out_path}")

    print("\nSummary:")
    for cam_id, (R, T) in cam_poses.items():
        C = -R.T @ T
        print(f"  cam {cam_id}: centre = "
              f"[{C[0,0]:.4f}, {C[1,0]:.4f}, {C[2,0]:.4f}]  "
              f"({cam_n_frames[cam_id]} frames)")

    print("\nStep 3 complete.")
    print("Next: fill in world_registration.csv then run Step 4.")


if __name__ == "__main__":
    main()
