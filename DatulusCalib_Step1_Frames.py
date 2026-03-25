#!/usr/bin/env python3
"""
Datalus Calibration — Step 1: Frame Extraction
All 8 videos (2 sessions × 4 cameras) are extracted in parallel.
Progress is shown as a live multi-bar display — one line per video.
"""

import os
import sys
import cv2
import time
import threading
import numpy as np
from pathlib import Path
import multiprocessing as mp

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SESSIONS = {
    "250707": {
        "102": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4",
        "108": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4",
        "113": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4",
        "117": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4",
    },
    "250708": {
        "102": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_102_20250708161657.mp4",
        "108": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_108_20250708161657.mp4",
        "113": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_113_20250708161657.mp4",
        "117": "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_117_20250708161657.mp4",
    },
}

CHESSBOARD        = (13, 9)
SQUARE_SIZE_MM    = 40.0
SQUARE_SIZE_M     = SQUARE_SIZE_MM / 1000.0

FRAME_INTERVAL_MS = 0          # 0 = every frame (no downsampling)

SENSOR_WIDTH_MM   = 11.2
FOCAL_LENGTH_MM   = 8.0

COLMAP_BIN        = "/opt/homebrew/bin/colmap"
OUTPUT_DIR        = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"
WORLD_CSV         = os.path.join(OUTPUT_DIR, "world_registration.csv")

PNG_PARAMS        = [cv2.IMWRITE_PNG_COMPRESSION, 0]  # fastest write, no compression
UPDATE_EVERY      = 200   # send a progress update every N frames
BAR_WIDTH         = 28

# ─── WORKER ───────────────────────────────────────────────────────────────────

_worker_queue = None   # set in each worker process via initializer

def _init_worker(q):
    global _worker_queue
    _worker_queue = q


def _extract_video(args):
    """
    Worker: extract all frames from one video.
    Sends (label, current_frame, total_frames, saved_count) to the shared queue.
    Uses cap.grab() (no decode) for skipped frames when step > 1.
    Returns (session, cam_id, w, h, saved).
    """
    session, cam_id, video_path, cam_dir_str, frame_interval_ms = args
    label   = f"{session}/cam{cam_id}"
    cam_dir = Path(cam_dir_str)
    cam_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _worker_queue.put((label, 0, 1, 0))
        return session, cam_id, None, None, -1

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step  = 1 if frame_interval_ms == 0 else max(1, int(round(fps * frame_interval_ms / 1000.0)))

    _worker_queue.put((label, 0, total, 0))

    idx = saved = 0
    while True:
        if idx % step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(cam_dir / f"{session}_frame_{idx:06d}.png"), frame, PNG_PARAMS)
            saved += 1
        else:
            if not cap.grab():
                break
        idx += 1
        if idx % UPDATE_EVERY == 0:
            _worker_queue.put((label, idx, total, saved))

    cap.release()
    _worker_queue.put((label, total, total, saved))
    return session, cam_id, w, h, saved


# ─── PROGRESS MONITOR ─────────────────────────────────────────────────────────

def _render_line(state, labels):
    """Single compact status line, rewritten in-place with \\r."""
    parts = []
    done  = 0
    for lbl in labels:
        cur, tot, sav = state[lbl]
        pct = int(min(1.0, cur / max(tot, 1)) * 100)
        if cur >= tot and tot > 1:
            parts.append(f"{lbl}:done")
            done += 1
        else:
            parts.append(f"{lbl}:{pct:3d}%")
    return f"  [{done}/{len(labels)}] " + "  ".join(parts)


def _drain(queue, state):
    while True:
        try:
            lbl, cur, tot, sav = queue.get_nowait()
            state[lbl] = (cur, tot, sav)
        except Exception:
            break


def _monitor(queue, labels, stop_event):
    """Monitor thread: rewrites a single status line every 150ms using \\r."""
    state = {lbl: (0, 1, 0) for lbl in labels}

    while not stop_event.is_set():
        _drain(queue, state)
        line = _render_line(state, labels)
        sys.stdout.write(f"\r{line}  ")
        sys.stdout.flush()
        time.sleep(0.15)

    # Final update
    _drain(queue, state)
    line = _render_line(state, labels)
    sys.stdout.write(f"\r{line}  \n")
    sys.stdout.flush()


# ─── STEP 1 ───────────────────────────────────────────────────────────────────

def extract_frames(sessions, output_root, frame_interval_ms=0):
    frames_root = Path(output_root) / "frames"

    jobs = []
    for session, camera_videos in sessions.items():
        for cam_id, video_path in camera_videos.items():
            if not os.path.exists(video_path):
                print(f"[ERROR] Video not found: {video_path}")
                sys.exit(1)
            jobs.append((session, cam_id, video_path,
                         str(frames_root / cam_id), frame_interval_ms))

    labels    = [f"{s}/cam{c}" for s, c, *_ in jobs]
    n_workers = min(len(jobs), mp.cpu_count())

    print(f"\n  {len(jobs)} videos — {n_workers} parallel workers")

    queue      = mp.Queue()
    stop_event = threading.Event()
    monitor    = threading.Thread(target=_monitor, args=(queue, labels, stop_event), daemon=True)
    monitor.start()

    image_size = {}
    with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(queue,)) as pool:
        for session, cam_id, w, h, saved in pool.imap_unordered(_extract_video, jobs):
            if saved == -1:
                stop_event.set()
                monitor.join()
                print(f"\n[ERROR] Cannot open video for [{session}] cam {cam_id}")
                sys.exit(1)
            if cam_id not in image_size and w is not None:
                image_size[cam_id] = (w, h)

    stop_event.set()
    monitor.join()

    return frames_root, image_size


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("DATALUS CALIBRATION — STEP 1: FRAME EXTRACTION")
    print("=" * 60)

    frames_root, image_size = extract_frames(SESSIONS, OUTPUT_DIR, FRAME_INTERVAL_MS)

    print("\nStep 1 complete.")
    print("Detected resolutions:")
    for cam_id, (w, h) in sorted(image_size.items()):
        f_px = FOCAL_LENGTH_MM * w / SENSOR_WIDTH_MM
        print(f"  cam {cam_id}: {w}x{h}  ->  prior focal = {f_px:.1f} px")


if __name__ == "__main__":
    main()