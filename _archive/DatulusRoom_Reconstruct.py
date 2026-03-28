#!/usr/bin/env python3
"""
Datalus Room Reconstruction
Runs COLMAP sparse + dense 3D reconstruction on a phone video of the home cage.

Usage:
    1. Drop your phone video (.mp4 / .mov / .avi) into VIDEO_DIR
    2. Run:  ./run.sh DatulusRoom_Reconstruct.py

Output is written to OUTPUT_DIR:
    sparse_txt/   — COLMAP sparse model (cameras, images, points3D)
    dense/        — dense point cloud (fused.ply) and mesh (meshed-poisson.ply)
                    if dense reconstruction succeeds (requires CUDA on Linux/Windows)

View results in MeshLab or CloudCompare (both free):
    brew install --cask meshlab
"""

import os
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

VIDEO_DIR         = "/Users/acalapai/ownCloud/Shared/PriCaB/HomeCage/Measurements"
OUTPUT_DIR        = "/Users/acalapai/ownCloud/Shared/PriCaB/HomeCage/Measurements/output"

FRAME_INTERVAL_MS = 500      # one frame every 500 ms (2 fps)
SEQUENTIAL_OVERLAP = 15      # match each frame against its 15 nearest neighbours
N_THREADS          = 12

PNG_PARAMS        = [cv2.IMWRITE_PNG_COMPRESSION, 0]
VIDEO_EXTENSIONS  = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def find_colmap() -> str:
    for c in ["/opt/homebrew/bin/colmap", "/usr/local/bin/colmap"]:
        if os.path.isfile(c):
            return c
    found = shutil.which("colmap")
    if found:
        return found
    print("[ERROR] colmap not found. Install with: brew install colmap")
    sys.exit(1)


def run(cmd, desc: str, optional: bool = False) -> bool:
    print(f"\n  >> {desc}")
    print(f"     {' '.join(str(c) for c in cmd)}")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        if optional:
            print(f"  [SKIP] {desc} failed (exit {result.returncode}) — continuing")
            return False
        print(f"\n[ERROR] {desc} failed (exit {result.returncode})")
        sys.exit(1)
    return True


def find_video(video_dir: str) -> Path:
    videos = sorted(
        p for p in Path(video_dir).iterdir()
        if p.suffix in VIDEO_EXTENSIONS and not p.name.startswith(".")
    )
    if not videos:
        print(f"[ERROR] No video files found in {video_dir}")
        print(f"  Drop a .mp4 / .mov file there and re-run.")
        sys.exit(1)
    if len(videos) == 1:
        return videos[0]
    print(f"\n  Multiple videos found:")
    for i, v in enumerate(videos):
        print(f"    [{i}] {v.name}")
    idx = int(input("  Select index: "))
    return videos[idx]


def extract_frames(video_path: Path, frames_dir: Path,
                   interval_ms: int) -> tuple:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step  = max(1, int(round(fps * interval_ms / 1000.0)))
    est   = total // step

    print(f"  {video_path.name}  |  {w}×{h}  {fps:.1f} fps  {total} frames")
    print(f"  Sampling every {step} frames ({interval_ms} ms)  →  ~{est} frames")

    idx = saved = 0
    while True:
        if idx % step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(frames_dir / f"frame_{idx:06d}.png"), frame, PNG_PARAMS)
            saved += 1
        else:
            if not cap.grab():
                break
        idx += 1
        if idx % 300 == 0:
            sys.stdout.write(
                f"\r  Extracting... {int(idx/max(total,1)*100):3d}%  saved={saved}")
            sys.stdout.flush()

    cap.release()
    print(f"\r  Extracted {saved} frames                          ")
    return saved, (w, h)


def best_model(sparse_dir: Path) -> Path:
    models = sorted(sparse_dir.iterdir())
    if not models:
        print("[ERROR] Mapper produced no models.")
        sys.exit(1)
    def n_images(m):
        t = m / "images.txt"
        if not t.exists():
            return 0
        return sum(1 for l in open(t) if not l.startswith("#") and l.strip()) // 2
    best = max(models, key=n_images)
    print(f"  Best model: {best.name}  ({n_images(best)} registered images)")
    return best


def print_summary(output_dir: Path, sparse_txt: Path):
    print(f"\n{'='*60}")
    print("  RECONSTRUCTION SUMMARY")
    print(f"{'='*60}")

    images_txt = sparse_txt / "images.txt"
    points_txt = sparse_txt / "points3D.txt"

    if images_txt.exists():
        n = sum(1 for l in open(images_txt) if not l.startswith("#") and l.strip()) // 2
        print(f"  Registered images : {n}")
    if points_txt.exists():
        n = sum(1 for l in open(points_txt) if not l.startswith("#") and l.strip())
        print(f"  Sparse 3D points  : {n:,}")

    fused = output_dir / "dense" / "fused.ply"
    mesh  = output_dir / "dense" / "meshed-poisson.ply"
    if fused.exists():
        print(f"  Dense cloud       : {fused}  ({fused.stat().st_size/1e6:.1f} MB)")
    if mesh.exists():
        print(f"  Mesh              : {mesh}  ({mesh.stat().st_size/1e6:.1f} MB)")

    print(f"\n  Output: {output_dir}")
    print(f"  View in MeshLab:  open {sparse_txt/'points3D.txt'}")
    if fused.exists():
        print(f"  or dense cloud:   open {fused}")
    print(f"{'='*60}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    _setup_log(str(Path(__file__).parent / "run.log"))
    print("=" * 60)
    print("DATALUS — ROOM RECONSTRUCTION")
    print("=" * 60)

    os.makedirs(VIDEO_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    colmap = find_colmap()
    print(f"\n  COLMAP: {colmap}")

    out        = Path(OUTPUT_DIR)
    frames_dir = out / "frames"
    sparse_dir = out / "sparse"
    sparse_txt = out / "sparse_txt"
    dense_dir  = out / "dense"
    db_path    = out / "database.db"

    for d in [sparse_dir, sparse_txt, dense_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Find video ────────────────────────────────────────────────────────────
    print(f"\n  Scanning {VIDEO_DIR} for videos...")
    video_path = find_video(VIDEO_DIR)
    print(f"  Using: {video_path.name}")

    # ── [1] Frame extraction ─────────────────────────────────────────────────
    print("\n[1] FRAME EXTRACTION")
    n_frames, _ = extract_frames(video_path, frames_dir, FRAME_INTERVAL_MS)

    # ── [2] Feature extraction ───────────────────────────────────────────────
    print("\n[2] FEATURE EXTRACTION")
    if db_path.exists():
        db_path.unlink()

    run([
        colmap, "feature_extractor",
        "--database_path",                 db_path,
        "--image_path",                    frames_dir,
        "--ImageReader.camera_model",      "SIMPLE_RADIAL",
        "--ImageReader.single_camera",     "1",      # all frames = same phone
        "--FeatureExtraction.use_gpu",     "0",
        "--FeatureExtraction.num_threads", N_THREADS,
    ], "Feature extraction")

    # ── [3] Sequential matching ──────────────────────────────────────────────
    # Sequential matcher is designed for video — far faster than exhaustive
    # and more accurate since it exploits temporal ordering.
    print("\n[3] SEQUENTIAL FEATURE MATCHING")
    run([
        colmap, "sequential_matcher",
        "--database_path",                     db_path,
        "--SequentialMatching.overlap",        SEQUENTIAL_OVERLAP,
        "--SequentialMatching.loop_detection", "1",
        "--FeatureMatching.use_gpu",           "0",
        "--FeatureMatching.num_threads",       N_THREADS,
    ], "Sequential matching")

    # ── [4] Sparse reconstruction ────────────────────────────────────────────
    print("\n[4] SPARSE RECONSTRUCTION")
    run([
        colmap, "mapper",
        "--database_path", db_path,
        "--image_path",    frames_dir,
        "--output_path",   sparse_dir,
    ], "Sparse reconstruction (mapper)")

    model = best_model(sparse_dir)

    run([
        colmap, "model_converter",
        "--input_path",  model,
        "--output_path", sparse_txt,
        "--output_type", "TXT",
    ], "Convert model to TXT")

    # ── [5] Dense reconstruction (optional — requires CUDA on Linux/Windows) ─
    # On Mac without CUDA this will likely fail gracefully.
    print("\n[5] DENSE RECONSTRUCTION (optional)")

    undistorted = out / "undistorted"
    ok = run([
        colmap, "image_undistorter",
        "--image_path",  frames_dir,
        "--input_path",  model,
        "--output_path", undistorted,
        "--output_type", "COLMAP",
    ], "Image undistortion", optional=True)

    if ok:
        ok = run([
            colmap, "patch_match_stereo",
            "--workspace_path",   undistorted,
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "false",
        ], "Patch match stereo", optional=True)

    if ok:
        ok = run([
            colmap, "stereo_fusion",
            "--workspace_path",   undistorted,
            "--workspace_format", "COLMAP",
            "--input_type",       "geometric",
            "--output_path",      str(dense_dir / "fused.ply"),
        ], "Stereo fusion → fused.ply", optional=True)

    if ok:
        run([
            colmap, "poisson_mesher",
            "--input_path",  str(dense_dir / "fused.ply"),
            "--output_path", str(dense_dir / "meshed-poisson.ply"),
        ], "Poisson meshing → mesh.ply", optional=True)

    if not ok:
        print("  Dense reconstruction skipped (no CUDA).")
        print("  You still have a sparse point cloud in sparse_txt/points3D.txt")
        print("  For dense results on Mac, consider: Meshroom or RealityCapture")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(out, sparse_txt)


if __name__ == "__main__":
    main()
