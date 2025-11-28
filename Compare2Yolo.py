#!/usr/bin/env python3
import os
import cv2
import time
from glob import glob
from ultralytics import YOLO
import re

# ============================================================
# CONFIG
# ============================================================
RAW_BASE = "/Users/acalapai/Desktop/Collage/RAW/250711/"         # unprocessed videos
PROC_BASE = "/Users/acalapai/Desktop/Collage"           # ABT-processed videos
OUT_DIR = "/Users/acalapai/Desktop/Collage/YOLOv9_Test"         # where collage videos go

MODEL_PATH = "yolov9c.pt"    # YOLOv9 checkpoint
PREVIEW_MIN = 5              # first 5 minutes
DOWNSAMPLE = 0.5             # scale factor for speed

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading YOLOv9 model…")
model = YOLO(MODEL_PATH)

# ============================================================
# CAMERA-ID EXTRACTION
# RAW videos look like: 102_20250708161657.mp4
# PROCESSED videos look like: July_2025__102_....
# ============================================================
def extract_cam_id(path):
    name = os.path.basename(path)

    # RAW style: 102_202507...
    m = re.match(r"(\d{3})_", name)
    if m:
        return m.group(1)

    # PROCESSED style: July_2025__102_....
    m = re.search(r"__([0-9]{3})_", name)
    if m:
        return m.group(1)

    return None

# ============================================================
# LOCATE VIDEOS
# ============================================================
raw_videos = sorted(glob(os.path.join(RAW_BASE, "*.mp4")))
proc_videos = sorted(glob(os.path.join(PROC_BASE, "*.mp4")))

if not raw_videos:
    raise RuntimeError(f"No RAW videos in {RAW_BASE}")
if not proc_videos:
    raise RuntimeError(f"No PROCESSED videos in {PROC_BASE}")

raw_dict = {extract_cam_id(v): v for v in raw_videos}
proc_dict = {extract_cam_id(v): v for v in proc_videos}

cams = sorted(set(raw_dict.keys()) & set(proc_dict.keys()))

if not cams:
    raise RuntimeError("No matching camera IDs between raw and processed videos.")

print("\nMatched Cameras:")
for c in cams:
    print(f"  Cam {c}:")
    print(f"     RAW:  {raw_dict[c]}")
    print(f"     PROC: {proc_dict[c]}")

# ============================================================
# PROGRESS BAR
# ============================================================
def bar(prefix, i, total):
    width = 30
    r = min(1.0, i / float(total))
    fill = int(r * width)
    line = "#" * fill + "-" * (width - fill)
    print(f"\r{prefix} [{line}] {i}/{total}", end="", flush=True)

# ============================================================
# PROCESS EACH CAMERA
# ============================================================
for cam in cams:

    raw_path = raw_dict[cam]
    proc_path = proc_dict[cam]

    print("\n=====================================================")
    print(f"Processing camera {cam}")
    print(f"RAW:       {raw_path}")
    print(f"PROCESSED: {proc_path}")
    print("=====================================================\n")

    cap_raw = cv2.VideoCapture(raw_path)
    cap_proc = cv2.VideoCapture(proc_path)

    if not cap_raw.isOpened():
        print("❌ ERROR: cannot open raw video")
        continue
    if not cap_proc.isOpened():
        print("❌ ERROR: cannot open processed video")
        continue

    fps = cap_raw.get(cv2.CAP_PROP_FPS)
    total_frames_raw  = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_proc = int(cap_proc.get(cv2.CAP_PROP_FRAME_COUNT))

    limit_frames = int(PREVIEW_MIN * 60 * fps)
    limit_frames = min(limit_frames, total_frames_raw, total_frames_proc)

    print(f"FPS = {fps}")
    print(f"RAW frames  = {total_frames_raw}")
    print(f"PROC frames = {total_frames_proc}")
    print(f"Processing first {limit_frames} frames\n")

    out_path = os.path.join(OUT_DIR, f"cam_{cam}_YOLOv9_compare.mp4")
    out_writer = None
    frame_idx = 0

    while frame_idx < limit_frames:

        bar("Progress", frame_idx, limit_frames)

        ok1, raw_frame = cap_raw.read()
        ok2, proc_frame = cap_proc.read()
        if not (ok1 and ok2):
            break

        # -------------------------------------------
        # Downsample (optional)
        # -------------------------------------------
        if DOWNSAMPLE != 1.0:
            raw_small = cv2.resize(raw_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            proc_small = cv2.resize(proc_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
        else:
            raw_small, proc_small = raw_frame, proc_frame

        # -------------------------------------------
        # YOLOv9 DETECTION
        # -------------------------------------------
        yres = model.predict(raw_small, device="mps", verbose=False)
        yolo_vis = yres[0].plot()   # overlayed bounding boxes

        # -------------------------------------------
        # CREATE SIDE-BY-SIDE VIEW
        # -------------------------------------------
        combined = cv2.hconcat([proc_small, yolo_vis])

        if out_writer is None:
            H, W = combined.shape[:2]
            out_writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (W, H)
            )

        out_writer.write(combined)
        frame_idx += 1

    print("\n")
    cap_raw.release()
    cap_proc.release()
    if out_writer:
        out_writer.release()

    print(f"✔ Saved: {out_path}\n")

print("\nDONE — all YOLOv9 comparisons created.")