#!/usr/bin/env python3
import os
import re
import json
import subprocess
from glob import glob

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

TOP_K = 10
MIN_CAMS_REQUIRED = 2
MIN_FRAME_GAP_SEC = 5          # <<< spread frames in time
SCALE_W = None
SCALE_H = None

# ============================================================
# Load TXT + MP4
# ============================================================
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))

def extract_cam_id(path):
    m = re.search(r"__([0-9]{3})_", os.path.basename(path))
    return m.group(1) if m else "???"

txt_by_cam = {extract_cam_id(t): t for t in TXT_FILES}
vid_by_cam = {extract_cam_id(v): v for v in VIDEOS}
cams = sorted(txt_by_cam.keys())

TXT_FILES = [txt_by_cam[c] for c in cams]
VIDEOS = [vid_by_cam[c] for c in cams]

# ============================================================
# FPS + resolution
# ============================================================
def probe_stream_info(path):
    info = json.loads(subprocess.check_output([
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams", path
    ]))
    return [s for s in info["streams"] if s["codec_type"] == "video"][0]

def get_fps(path):
    s = probe_stream_info(path)
    num, den = map(float, s["r_frame_rate"].split("/"))
    return num / den

def probe_resolution(path):
    s = probe_stream_info(path)
    return int(s["width"]), int(s["height"])

FPS = get_fps(VIDEOS[0])
MIN_FRAME_GAP = int(MIN_FRAME_GAP_SEC * FPS)

sizes = [probe_resolution(v) for v in VIDEOS]
W = min(w for w, _ in sizes)
H = min(h for _, h in sizes)

# ============================================================
# Parse TXT → centers
# ============================================================
def load_centers(path):
    rows = []
    with open(path) as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())
            if len(parts) < 8 or not parts[0].isdigit():
                continue
            frame = int(parts[0])
            monkey = parts[1]
            try:
                x1, y1, x2, y2 = map(float, parts[2:6])
            except:
                continue
            rows.append((frame, monkey, (x1 + x2) / 2, (y1 + y2) / 2))
    return pd.DataFrame(rows, columns=["frame", "monkey", "cx", "cy"])

def distance_for_camera(txt_path):
    df = load_centers(txt_path)
    pivot = df.pivot_table(index="frame", columns="monkey", values=["cx", "cy"], aggfunc="first")
    try:
        dx = pivot["cx"][MONKEY_A] - pivot["cx"][MONKEY_B]
        dy = pivot["cy"][MONKEY_A] - pivot["cy"][MONKEY_B]
    except KeyError:
        return pd.Series(dtype=float)
    return np.sqrt(dx**2 + dy**2)

# ============================================================
# Build multiview distance table
# ============================================================
dist_series = []
for t in TXT_FILES:
    s = distance_for_camera(t)
    s.name = extract_cam_id(t)
    dist_series.append(s)

df_dist = pd.concat(dist_series, axis=1).sort_index()
valid = df_dist.notna().sum(axis=1) >= MIN_CAMS_REQUIRED
min_dist = df_dist.min(axis=1).where(valid).dropna()

# ============================================================
# Temporal diversification (THIS IS THE FIX)
# ============================================================
selected = []
last_selected = -np.inf

for frame, d in min_dist.sort_values().items():
    if frame - last_selected >= MIN_FRAME_GAP:
        selected.append((frame, d))
        last_selected = frame
    if len(selected) == TOP_K:
        break

selected_frames = pd.Series(
    [d for _, d in selected],
    index=[f for f, _ in selected]
)

print("\nSelected frames (spread in time):")
for f, d in selected_frames.items():
    print(f"  frame {f:6d} → {d:7.2f}px")

# ============================================================
# Distance distribution
# ============================================================
plt.figure(figsize=(10, 4))
plt.hist(min_dist.values, bins=60)
plt.xlabel("Min inter-monkey distance per multiview frame (px)")
plt.ylabel("Frame count")
plt.title("Distribution of inter-monkey distances (true multiview)")
plt.tight_layout()
plt.show()

# ============================================================
# Frame extraction
# ============================================================
def grab_frame(video, frame):
    sec = frame / FPS
    raw = subprocess.check_output([
        FFMPEG, "-ss", str(sec), "-i", video,
        "-vf", f"scale={W}:{H}",
        "-vframes", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ])
    return np.frombuffer(raw, np.uint8).reshape((H, W, 3))

def make_collage(frame):
    imgs = [grab_frame(v, frame) for v in VIDEOS]
    imgs = [cv2.copyMakeBorder(i, 10, 10, 10, 10, cv2.BORDER_CONSTANT) for i in imgs]
    return np.vstack([np.hstack(imgs[:2]), np.hstack(imgs[2:])])

# ============================================================
# Show 10 closest, temporally spread frames
# ============================================================
cols = 5
rows = int(np.ceil(TOP_K / cols))
fig, axes = plt.subplots(rows, cols, figsize=(18, 7))
axes = axes.flatten()

for ax, (frame, d) in zip(axes, selected_frames.items()):
    collage = make_collage(frame)
    ax.imshow(collage)
    ax.set_title(f"frame {frame} | d={d:.1f}px", fontsize=10)
    ax.axis("off")

for ax in axes[len(selected_frames):]:
    ax.axis("off")

plt.suptitle(
    f"Top {TOP_K} closest Elm–Jok multiview frames\n(min gap {MIN_FRAME_GAP_SEC}s)",
    y=1.02
)
plt.tight_layout()
plt.show()