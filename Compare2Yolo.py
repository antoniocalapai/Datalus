#!/usr/bin/env python3
import os
import cv2
import time
from glob import glob
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
RAW_BASE = "/Users/acalapai/Desktop/RawVideos"
PROC_BASE = "/Users/acalapai/Desktop/ProcessedVideos"
OUT_DIR = "/Users/acalapai/Desktop/YOLOv9_Comparisons"

MODEL_PATH = "yolov9c.pt"       # you downloaded this
PREVIEW_MIN = 5                 # first 5 minutes
DOWNSAMPLE = 0.5                # 50% reduction for speed
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
print("\nLoading YOLOv9 model…")
model = YOLO(MODEL_PATH)

# ============================================================
# FIND RAW & PROCESSED VIDEOS (match by camera ID)
# ============================================================
def extract_cam_id(path):
    """Matches your naming: __102_, __108_ etc."""
    import re
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else None

raw_videos = sorted(glob(os.path.join(RAW_BASE, "*.mp4")))
proc_videos = sorted(glob(os.path.join(PROC_BASE, "*.mp4")))

if len(raw_videos) == 0:
    raise RuntimeError("No RAW videos found.")
if len(proc_videos) == 0:
    raise RuntimeError("No PROCESSED videos found.")

raw_dict = {extract_cam_id(v): v for v in raw_videos}
proc_dict = {extract_cam_id(v): v for v in proc_videos}

# Keep only cameras present in both
cams = sorted(set(raw_dict.keys()) & set(proc_dict.keys()))
if len(cams) == 0:
    raise RuntimeError("No matching camera IDs found across raw + processed videos.")

print("\nMatched cameras:")
for c in cams:
    print(f"  Cam {c}:")
    print(f"       RAW:  {raw_dict[c]}")
    print(f"       PROC: {proc_dict[c]}")

# ============================================================
# MAIN LOOP FOR EACH CAMERA
# ============================================================
for cam in cams:

    raw_path = raw_dict[cam]
    proc_path = proc_dict[cam]

    print("\n=====================================================")
    print(f"Processing camera {cam}")
    print("RAW:       ", raw_path)
    print("PROCESSED: ", proc_path)
    print("=====================================================\n")

    cap_raw = cv2.VideoCapture(raw_path)
    cap_proc = cv2.VideoCapture(proc_path)

    if not cap_raw.isOpened():
        print("ERROR: cannot open raw video:", raw_path)
        continue
    if not cap_proc.isOpened():
        print("ERROR: cannot open processed video:", proc_path)
        continue

    fps = cap_raw.get(cv2.CAP_PROP_FPS)
    total_frames = cap_raw.get(cv2.CAP_PROP_FRAME_COUNT)
    limit_frames = int(PREVIEW_MIN * 60 * fps)

    print(f"FPS = {fps}")
    print(f"Total frames raw  = {int(total_frames)}")
    print(f"Processing up to   = {limit_frames}\n")

    # Output path
    out_path = os.path.join(OUT_DIR, f"cam_{cam}_YOLOv9_compare.mp4")

    # Video writer
    out_w = None
    out_h = None
    out_writer = None

    frame_idx = 0

    while True:
        ok1, raw_frame = cap_raw.read()
        ok2, proc_frame = cap_proc.read()

        if not ok1 or not ok2:
            print("Reached end of one of the videos.")
            break

        if frame_idx >= limit_frames:
            print("Reached preview limit.")
            break

        # -------------------------------------------
        # Downsample both views
        # -------------------------------------------
        if DOWNSAMPLE != 1.0:
            raw_small = cv2.resize(raw_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            proc_small = cv2.resize(proc_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
        else:
            raw_small, proc_small = raw_frame, proc_frame

        # -------------------------------------------
        # YOLOv9 DETECTION
        # -------------------------------------------
        results = model.predict(
            raw_small,
            device="mps",
            verbose=False
        )
        yolo_vis = results[0].plot()   # detections drawn

        # -------------------------------------------
        # SIDE-BY-SIDE COLLAGE
        # left  = processed ABT video
        # right = YOLOv9 detection
        # -------------------------------------------
        combined = cv2.hconcat([proc_small, yolo_vis])

        # initialize writer once
        if out_writer is None:
            out_h, out_w = combined.shape[:2]
            out_writer = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out_w, out_h)
            )

        out_writer.write(combined)
        frame_idx += 1

    cap_raw.release()
    cap_proc.release()
    if out_writer:
        out_writer.release()

    print(f"Saved: {out_path}")

print("\nDONE — all comparisons complete.")