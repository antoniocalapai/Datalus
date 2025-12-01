#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
from matplotlib.gridspec import GridSpec

# ================================================================
# GLOBAL CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage"
FPS = 5
save_plots = True

plot_dir = os.path.join(BASE, "plots")
os.makedirs(plot_dir, exist_ok=True)

sns.set_theme(style="whitegrid")

# Keypoint names in ABT order
KP_NAMES = [
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# ================================================================
# LOAD TXT FILES
# ================================================================
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))
if not TXT_FILES:
    raise RuntimeError("No ABT txt files found.")

print("\nUsing TXT files:")
for t in TXT_FILES:
    print(" -", t)


# ================================================================
# HELPERS
# ================================================================
def find_processed_mp4(txt_path):
    """
    ABT result MP4 has same camera ID but different ending timestamp.
    """
    name = os.path.basename(txt_path)
    m = re.search(r"__(\d{3})_", name)
    if not m:
        raise RuntimeError(f"Cannot extract camera ID from {txt_path}")
    cam = m.group(1)

    pattern = os.path.join(
        os.path.dirname(txt_path),
        f"July_2025__{cam}_*_2D_result.mp4"
    )
    files = glob(pattern)
    if not files:
        raise RuntimeError(f"No processed MP4 found for camera {cam}")
    return files[0]


def get_frame_count(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ================================================================
# PARSE ALL KEYPOINTS (TEMP STRUCTURE)
# ================================================================
temp = []
cam_video = {}   # cam_id → (mp4_path, total_frames)

for path in TXT_FILES:
    cam_id = re.findall(r"__(\d{3})_", os.path.basename(path))[0]

    mp4_path = find_processed_mp4(path)
    total_frames = get_frame_count(mp4_path)
    cam_video[cam_id] = (mp4_path, total_frames)

    with open(path, "r") as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())

            # skip header
            if parts[0] == "frame_number":
                continue
            if len(parts) < 10:
                continue

            try:
                frame = int(parts[0])
            except Exception:
                continue

            # keypoints start at index 7
            kp_vals = parts[7:]
            K = len(kp_vals) // 3

            kpx, kpy = [], []
            for i in range(0, len(kp_vals), 3):
                try:
                    x = float(kp_vals[i])
                    y = float(kp_vals[i + 1])
                except Exception:
                    x, y = np.nan, np.nan
                kpx.append(x)
                kpy.append(y)

            temp.append([cam_id, frame, total_frames, K, kpx, kpy])

temp_df = pd.DataFrame(
    temp, columns=["camera", "frame", "total", "K", "kpx", "kpy"]
)

# ================================================================
# BUILD FULL TIMELINES (NaN-FILLED)
# ================================================================
records = []

for cam_id, sub in temp_df.groupby("camera"):
    total_frames = sub["total"].iloc[0]
    K = sub["K"].iloc[0]

    full_x = np.full((total_frames, K), np.nan)
    full_y = np.full((total_frames, K), np.nan)

    for _, row in sub.iterrows():
        f = row["frame"]
        if f < total_frames:
            full_x[f] = row["kpx"]
            full_y[f] = row["kpy"]

    for f in range(total_frames):
        records.append([cam_id, f, K, list(full_x[f]), list(full_y[f])])

df = pd.DataFrame(records, columns=["camera", "frame", "n_kp", "kpx", "kpy"])

print("\nFull reconstructed dataframe:")
print(df.head())

# ================================================================
# COMPUTE JITTER
# ================================================================
jitter_time_rows = []
summary_rows = []

for cam_id, subdf in df.groupby("camera"):
    subdf = subdf.sort_values("frame")
    K = subdf.iloc[0]["n_kp"]

    xmat = np.vstack(subdf["kpx"].values)
    ymat = np.vstack(subdf["kpy"].values)

    dx = xmat[1:] - xmat[:-1]
    dy = ymat[1:] - ymat[:-1]
    dist = np.sqrt(dx**2 + dy**2)  # (T-1, K)

    # summary per keypoint (mean jitter)
    for k in range(K):
        summary_rows.append([cam_id, k, np.nanmean(dist[:, k])])

    # per-frame jitter rows (keep NaNs for gaps)
    T = dist.shape[0]
    for t in range(T):
        for k in range(K):
            jitter_time_rows.append([cam_id, t, k, dist[t, k]])

jdf = pd.DataFrame(summary_rows, columns=["camera", "keypoint", "mean_jitter"])
jdf["kp_name"] = jdf["keypoint"].apply(lambda i: KP_NAMES[i])

df_time = pd.DataFrame(
    jitter_time_rows, columns=["camera", "t", "keypoint", "jitter"]
)
df_time["kp_name"] = df_time["keypoint"].apply(lambda i: KP_NAMES[i])

# smoothed version for time plots
df_time["jitter_smooth"] = (
    df_time.groupby(["camera", "keypoint"])["jitter"]
    .transform(lambda x: x.rolling(window=7, center=True, min_periods=1).median())
)

cams_sorted = sorted(df_time["camera"].unique())
n_cams = len(cams_sorted)

# global x-limit for boxplots (clip extreme outliers at 99th percentile)
global_max = np.nanpercentile(df_time["jitter"], 99)
if not np.isfinite(global_max) or global_max <= 0:
    global_max = None  # fallback: let seaborn choose

# ================================================================
# MASTER FIGURE (JITTER + BOXPLOTS + REPRESENTATIVE FRAME)
# ================================================================
# Roughly MacBook screen size
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(
    n_cams,
    3,
    width_ratios=[1.8, 1.5, 1.1],
    figure=fig,
)

for row_idx, cid in enumerate(cams_sorted):
    dfc = df_time[df_time["camera"] == cid]

    # ---------------- LEFT: JITTER OVER TIME ----------------
    ax_plot = fig.add_subplot(gs[row_idx, 0])

    for kp in KP_NAMES:
        sub = dfc[dfc["kp_name"] == kp].sort_values("t")
        ax_plot.plot(sub["t"], sub["jitter_smooth"], linewidth=0.4)

    ax_plot.set_title(f"Camera {cid}", fontsize=9)
    ax_plot.set_ylabel("jitter (px)", fontsize=8)
    ax_plot.tick_params(axis="both", labelsize=7)

    if row_idx == n_cams - 1:
        ax_plot.set_xlabel("frame index", fontsize=8)
    else:
        ax_plot.set_xlabel("")

    # ---------------- MIDDLE: BOXPLOTS (per keypoint) ----------------
    ax_box = fig.add_subplot(gs[row_idx, 1])

    dfb = dfc.copy()

    sns.boxplot(
        data=dfb,
        x="jitter",
        y="kp_name",
        orient="h",
        showcaps=True,
        showfliers=False,
        width=0.6,
        ax=ax_box,
    )

    # put kp labels on the right side
    ax_box.yaxis.tick_right()
    ax_box.yaxis.set_label_position("right")
    ax_box.set_ylabel("")
    ax_box.set_xlabel("jitter (px)", fontsize=8)
    ax_box.set_title("keypoint jitter distribution", fontsize=8)
    ax_box.tick_params(axis="y", labelsize=6)
    ax_box.tick_params(axis="x", labelsize=7)
    ax_box.set_xlim(0, dfb["jitter"].quantile(0.99))

    if global_max is not None:
        ax_box.set_xlim(0, global_max)

    # ---------------- RIGHT: REPRESENTATIVE FRAME ----------------
    ax_img = fig.add_subplot(gs[row_idx, 2])

    per_frame_j = dfc.groupby("t")["jitter_smooth"].mean()
    per_frame_j = per_frame_j.replace([np.inf, -np.inf], np.nan)

    if per_frame_j.dropna().empty:
        best_frame = 0
    else:
        best_frame = int(per_frame_j.idxmax())

    vid_path, total_frames = cam_video[cid]
    cap = cv2.VideoCapture(vid_path)
    best_frame_clamped = max(0, min(best_frame, total_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_frame_clamped)
    ok, frame = cap.read()
    cap.release()

    if ok and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

    ax_img.imshow(frame_rgb)
    ax_img.set_title(f"frame {best_frame_clamped}", fontsize=8)
    ax_img.axis("off")

plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
plt.show()

if save_plots:
    out_path = os.path.join(plot_dir, "jitter_full_master_boxplots.png")
    fig.savefig(out_path, dpi=200)
    print("Saved figure:", out_path)

print("\nDone.\n")