import os
import re
import cv2
import shutil
from glob import glob
import pandas as pd
import numpy as np
import deeplabcut

# ============================================================
# CONFIG
# ============================================================
RAW_BASE = "/Users/acalapai/Desktop/Collage/RAW/250711"
PROC_BASE = "/Users/acalapai/Desktop/Collage"
OUT_DIR   = "/Users/acalapai/Desktop/Collage/DLC_Test"

MODEL = "full_macaque"
PROJECT_NAME = "DLC_FullBody_Compare"
USER = "anc"

PREVIEW_MIN = 5          # minutes
DOWNSAMPLE = 0.5         # for comparison video

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# CAMERA ID extraction (same as Compare2Yolo)
# ============================================================
def extract_cam_id(path):
    name = os.path.basename(path)
    m = re.match(r"(\d{3})_", name)
    if m: return m.group(1)
    m = re.search(r"__([0-9]{3})_", name)
    if m: return m.group(1)
    return None


# ============================================================
# Manual DLC labeled video generator (macOS safe)
# ============================================================
def draw_dlc_labeled_video(video_path, csv_path, out_path, dotsize=4, pcutoff=0.1):

    df = pd.read_csv(csv_path, header=[0,1,2])

    # ----------------------------
    # AUTO-DETECT SCORER PREFIX
    # ----------------------------
    scorers = df.columns.get_level_values(0).unique().tolist()
    scorers = [s for s in scorers if "DLC" in s]  # keep only DLC scorers
    if len(scorers) == 0:
        raise RuntimeError(f"No DLC scorer found in CSV columns: {df.columns[:5]}")

    scorer = scorers[0]
    print(f"✔ Using scorer: {scorer}")

    # ----------------------------
    # AUTO-DETECT BODY PARTS
    # ----------------------------
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    bodyparts = [bp for bp in bodyparts if bp not in ["x", "y", "likelihood"]]
    print(f"✔ Bodyparts detected: {bodyparts}")

    # ----------------------------
    # Load video + prepare writer
    # ----------------------------
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw each bodypart
        for bp in bodyparts:
            try:
                x = df[(scorer, bp, "x")].iloc[frame_idx]
                y = df[(scorer, bp, "y")].iloc[frame_idx]
                p = df[(scorer, bp, "likelihood")].iloc[frame_idx]
            except KeyError:
                continue  # Skip missing bodypart

            if p > pcutoff and x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), dotsize, (0,255,0), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"✔ Manual labeled video saved to: {out_path}")


# ============================================================
# Locate RAW + processed videos
# ============================================================
raw_videos  = sorted(glob(os.path.join(RAW_BASE, "*.mp4")))
proc_videos = sorted(glob(os.path.join(PROC_BASE, "*.mp4")))

raw_dict  = {extract_cam_id(v): v for v in raw_videos}
proc_dict = {extract_cam_id(v): v for v in proc_videos}

cams = sorted(set(raw_dict) & set(proc_dict))
print("\nMatched cameras:", cams)


# ============================================================
# MAIN LOOP
# ============================================================
for cam in cams:

    print("\n=====================================================")
    print(f"Processing camera {cam}")
    print("=====================================================\n")

    raw_path  = raw_dict[cam]
    proc_path = proc_dict[cam]

    # -----------------------------------------
    # 1. Determine 5-min cut length
    # -----------------------------------------
    cap_raw = cv2.VideoCapture(raw_path)
    cap_proc = cv2.VideoCapture(proc_path)

    fps_raw = cap_raw.get(cv2.CAP_PROP_FPS)
    total_raw = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    total_proc = int(cap_proc.get(cv2.CAP_PROP_FRAME_COUNT))

    limit = int(PREVIEW_MIN * 60 * fps_raw)
    limit = min(limit, total_raw, total_proc)

    cap_raw.release()
    cap_proc.release()

    # -----------------------------------------
    # 2. Extract 5-min clip
    # -----------------------------------------
    temp_cut = os.path.join(OUT_DIR, f"cam_{cam}_cut_raw.mp4")

    cap = cv2.VideoCapture(raw_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    wtr = cv2.VideoWriter(temp_cut, fourcc, fps_raw, (w,h))

    idx = 0
    while idx < limit:
        ok, frame = cap.read()
        if not ok:
            break
        wtr.write(frame)
        idx += 1

    cap.release()
    wtr.release()
    print("✔ Created 5-min cut:", temp_cut)

    # -----------------------------------------
    # 3. DLC Downsample
    # -----------------------------------------
    print("Downsampling via DLC…")
    videotype = "mp4"
    video_down = deeplabcut.DownSampleVideo(temp_cut, width=900)

    # -----------------------------------------
    # 4. DLC pretrained inference
    # -----------------------------------------
    print("Creating DLC project…")
    cam_project = f"{PROJECT_NAME}_cam{cam}"

    cfg_path, train_cfg_path = deeplabcut.create_pretrained_project(
        project=cam_project,
        experimenter=USER,
        videos=[video_down],
        videotype=videotype,
        model=MODEL,
        analyzevideo=True,
        createlabeledvideo=False,
        copy_videos=True,
        working_directory=OUT_DIR
    )

    # Edit config (optional)
    deeplabcut.auxiliaryfunctions.edit_config(cfg_path, {
        'dotsize': 3,
        'pcutoff': 0.1,
    })

    project_path = os.path.dirname(cfg_path)
    video_in_project = os.path.join(project_path, "videos", os.path.basename(video_down))

    # -----------------------------------------
    # 5. Run prediction filtering (ignored later)
    # -----------------------------------------
    deeplabcut.filterpredictions(cfg_path, [video_in_project], videotype=videotype)

    # -----------------------------------------
    # 6. Load DLC output CSV
    #    (DLC writes CSV into videos/ folder)
    # -----------------------------------------
    csv_list = glob(os.path.join(project_path, "videos", "*.csv"))
    if len(csv_list) == 0:
        raise RuntimeError("❌ No DLC CSV found — prediction failed.")
    csv_path = csv_list[0]

    # -----------------------------------------
    # 7. Build labeled video manually (robust)
    # -----------------------------------------
    labeled_video = os.path.join(project_path, "manual_labeled.mp4")
    draw_dlc_labeled_video(video_in_project, csv_path, labeled_video)
    print("✔ Manual labeled video:", labeled_video)

    # -----------------------------------------
    # 8. Compare with ABT processed video
    # -----------------------------------------
    print("Building comparison video…")

    cap_proc = cv2.VideoCapture(proc_path)
    cap_dlc = cv2.VideoCapture(labeled_video)

    out_path = os.path.join(OUT_DIR, f"cam_{cam}_DLC_compare.mp4")
    out = None

    for i in range(limit):
        ok1, proc_frame = cap_proc.read()
        ok2, dlc_frame = cap_dlc.read()
        if not ok1 or not ok2:
            break

        if DOWNSAMPLE != 1.0:
            proc_frame = cv2.resize(proc_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            dlc_frame  = cv2.resize(dlc_frame,  None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)

        combined = cv2.hconcat([proc_frame, dlc_frame])

        if out is None:
            H, W = combined.shape[:2]
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_raw, (W, H))

        out.write(combined)

    cap_proc.release()
    cap_dlc.release()
    if out:
        out.release()

    print("✔ Saved comparison video:", out_path)

print("\nDONE — all cameras processed.")