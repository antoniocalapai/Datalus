#!/usr/bin/env python3
import os
import re
import json
import subprocess
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# ============================================================
# CONFIG
# ============================================================
BASE = "/Users/acalapai/Desktop/Collage"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"

MONKEY_A = "Elm"
MONKEY_B = "Jok"

# ============================================================
# Load TXT + MP4
# ============================================================
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))

if len(TXT_FILES) != 4:
    raise RuntimeError("Need 4 TXT files.")
if len(VIDEOS) != 4:
    raise RuntimeError("Need 4 MP4 files.")

def extract_cam_id(path):
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else "???"

def extract_start_timestamp(path):
    m = re.findall(r"(\d{14})", os.path.basename(path))
    return m[0] if m else None

# ============================================================
# FPS from video
# ============================================================
def probe_stream_info(path):
    info = json.loads(subprocess.check_output([
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]))
    return [s for s in info["streams"] if s["codec_type"] == "video"][0]

def get_fps(path):
    s = probe_stream_info(path)
    num, den = map(float, s["r_frame_rate"].split("/"))
    return num / den

FPS = get_fps(VIDEOS[0])

# ============================================================
# Resolution
# ============================================================
def probe_resolution(path):
    s = probe_stream_info(path)
    return s["width"], s["height"]

sizes = [probe_resolution(v) for v in VIDEOS]
W = min(s[0] for s in sizes)
H = min(s[1] for s in sizes)

# ============================================================
# Parse TXT → distances
# ============================================================
def load_centers(path):
    df = pd.read_csv(path, sep=r"\s+")
    # use only first bbox
    x1, y1, x2, y2 = df.iloc[:, 2], df.iloc[:, 3], df.iloc[:, 4], df.iloc[:, 5]
    return pd.DataFrame({
        "frame": df.iloc[:, 0],
        "monkey": df.iloc[:, 1],
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
    })

def distance_for_camera(path):
    centers = load_centers(path)
    pivot = centers.pivot_table(
        index="frame", columns="monkey", values=["cx", "cy"], aggfunc="first"
    )
    try:
        ax, ay = pivot["cx"][MONKEY_A], pivot["cy"][MONKEY_A]
        bx, by = pivot["cx"][MONKEY_B], pivot["cy"][MONKEY_B]
    except KeyError:
        return pd.Series(dtype=float)

    dx = ax - bx
    dy = ay - by
    dist = (dx**2 + dy**2) ** 0.5
    dist = dist.where(~dist.isna(), np.nan)

    cam_id = extract_cam_id(path)
    dist.name = f"Cam_{cam_id}"
    return dist

# Compute distances
dists = {extract_cam_id(t): distance_for_camera(t) for t in TXT_FILES}
df_dist = pd.concat(dists.values(), axis=1)

# Only frames seen in ≥2 cameras
valid = df_dist.notna().sum(axis=1) >= 2

# Combined minimum
min_dist = df_dist.min(axis=1).where(valid).dropna()

best_frame = min_dist.idxmin()
best_value = min_dist.loc[best_frame]

print("Best frame:", best_frame, "distance:", best_value)

# ============================================================
# Extract frame from each video
# ============================================================
def grab_frame(video, frame_num, W, H):
    sec = frame_num / FPS
    raw = subprocess.check_output([
        FFMPEG, "-ss", str(sec), "-i", video,
        "-vf", f"scale={W}:{H}",
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ])
    arr = np.frombuffer(raw, np.uint8).reshape((H, W, 3))
    return arr

frames = [grab_frame(v, best_frame, W, H) for v in VIDEOS]

# Build collage
border = 10
views = [cv2.copyMakeBorder(f, border, border, border, border, cv2.BORDER_CONSTANT) for f in frames]
top = np.hstack(views[:2])
bot = np.hstack(views[2:])
collage = np.vstack([top, bot])

# Show
plt.figure(figsize=(10, 8))
plt.imshow(collage)
plt.axis("off")
plt.title(f"Closest frame = {best_frame}  |  dist = {best_value:.1f}px")
plt.show()