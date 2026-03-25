import subprocess
import numpy as np
import cv2
import os
import re
from glob import glob
from collections import defaultdict

BASE = "/Users/acalapai/Desktop/Collage"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
TOP_N = 100
FPS = 5

# --------------------------------------------------------
# Detect videos + TXT
# --------------------------------------------------------
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))

if len(VIDEOS) != 4:
    raise RuntimeError(f"Expected 4 videos, found {len(VIDEOS)}")

if len(TXT_FILES) != 4:
    raise RuntimeError(f"Expected 4 TXT files, found {len(TXT_FILES)}")

print("\nUsing videos + TXT:")
for v in VIDEOS:
    print(" -", v)
for t in TXT_FILES:
    print(" -", t)

# --------------------------------------------------------
# Extract camera IDs + session date
# --------------------------------------------------------
def extract_cam_id(path):
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1)

def extract_date(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{8})\d{6}", name)
    return m.group(1)

CAMERA_IDS   = [extract_cam_id(v) for v in VIDEOS]
CAMERA_DATES = [extract_date(v) for v in VIDEOS]
SESSION_DATE = CAMERA_DATES[0]

print("\nDetected camera IDs:", CAMERA_IDS)
print("Detected session date:", SESSION_DATE)

# --------------------------------------------------------
# Analyze TXT files (count boxes per frame per camera)
# --------------------------------------------------------
frame_bbox_counts = defaultdict(lambda: [0,0,0,0])

for cam_idx, path in enumerate(TXT_FILES):
    with open(path, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
            except ValueError:
                # header or malformed line -> skip
                continue
            frame_bbox_counts[frame][cam_idx] += 1

frame_total = {frame: sum(counts) for frame, counts in frame_bbox_counts.items()}

# --------------------------------------------------------
# TOP-N frames
# --------------------------------------------------------
sorted_frames = sorted(frame_total.items(), key=lambda x: x[1], reverse=True)
frames_to_extract = [f for f, c in sorted_frames[:TOP_N]]

print(f"\nTop {TOP_N} frames identified.")
print("Max total boxes in any frame:", sorted_frames[0][1])

# --------------------------------------------------------
# Extract & build collages
# --------------------------------------------------------
extract_dir = os.path.join(BASE, "extracted_frames")
collage_dir = os.path.join(BASE, "collages")
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(collage_dir, exist_ok=True)

def extract_single_frame(video, frame_idx, out_path, W, H):
    timestamp = frame_idx / FPS
    cmd = [
        FFMPEG, "-loglevel", "quiet", "-y",
        "-ss", f"{timestamp}",
        "-i", video,
        "-vf", f"scale={W}:{H}",
        "-frames:v", "1",
        "-vsync", "0",
        "-q:v", "1",
        out_path
    ]
    subprocess.run(cmd)

# Find canonical resolution (same logic as script 1)
def probe_resolution(path):
    import json
    cmd = ["/opt/homebrew/bin/ffprobe", "-v", "quiet",
           "-print_format", "json", "-show_streams", path]
    info = json.loads(subprocess.check_output(cmd))
    stream = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    return stream["width"], stream["height"]

sizes = [probe_resolution(v) for v in VIDEOS]
W = min(s[0] for s in sizes)
H = min(s[1] for s in sizes)

# --------------------------------------------------------
# Extract frames with PROGRESS BAR
# --------------------------------------------------------
from tqdm import tqdm

print(f"\nExtracting {len(frames_to_extract)} frames...")

for frame_idx in tqdm(frames_to_extract, desc="Extracting & Collaging"):
    per_cam_paths = []

    # Extract per-camera frames
    for cam_idx, video in enumerate(VIDEOS):
        cam_id  = CAMERA_IDS[cam_idx]
        date    = CAMERA_DATES[cam_idx]

        out_png = os.path.join(
            extract_dir,
            f"{cam_id}_{date}_frame_{frame_idx:06d}.png"
        )

        extract_single_frame(video, frame_idx, out_png, W, H)
        per_cam_paths.append(out_png)

    # Load extracted frames
    imgs = [cv2.imread(p) for p in per_cam_paths]
    if any(i is None for i in imgs):
        # If one camera is missing this frame, just skip
        continue

    # Add border for Photoshop
    bordered = [
        cv2.copyMakeBorder(img, 20, 20, 20, 20,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
        for img in imgs
    ]

    # Build the 2×2 collage
    top = np.hstack(bordered[:2])
    bottom = np.hstack(bordered[2:])
    collage = np.vstack([top, bottom])

    collage_path = os.path.join(
        collage_dir,
        f"collage_{SESSION_DATE}_frame_{frame_idx:06d}.png"
    )

    cv2.imwrite(collage_path, collage)

print("\nDONE — frames + collages ready.")
print(f"Extracted frames in: {extract_dir}")
print(f"Collages in: {collage_dir}")

# --------------------------------------------------------
# SEABORN STACKED HISTOGRAM — boxes per frame per monkey
# --------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plot_dir = os.path.join(BASE, "plots")
os.makedirs(plot_dir, exist_ok=True)

boxes_per_monkey = defaultdict(lambda: defaultdict(int))

# Parse TXT files
for path in TXT_FILES:
    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue

            if not parts[0].isdigit():
                continue

            frame_idx = int(parts[0])
            monkey_id = parts[1]

            values = parts[2:]
            n_boxes = len(values) // 6

            boxes_per_monkey[monkey_id][frame_idx] += n_boxes

# DataFrame
rows = []
for m_id, frame_dict in boxes_per_monkey.items():
    for f, count in frame_dict.items():
        rows.append({"frame": f, "monkey": m_id, "count": count})

df = pd.DataFrame(rows)

sns.set_theme(style="whitegrid")

plt.figure(figsize=(16, 6))
sns.histplot(
    df,
    x="frame",
    weights="count",
    hue="monkey",
    bins=200,
    multiple="stack",
    palette="tab10",
)

plt.title(f"Stacked Histogram: Boxes per Frame per Monkey ({SESSION_DATE})", fontsize=16)
plt.xlabel("Frame index", fontsize=14)
plt.ylabel("Number of boxes", fontsize=14)

plt.tight_layout()

out_path = os.path.join(plot_dir, "stacked_hist_boxes_per_monkey.png")
plt.savefig(out_path, dpi=150)

# >>> SHOW THE PLOT HERE <<<
plt.show()

plt.close()

print(f"Saved stacked histogram: {out_path}")

