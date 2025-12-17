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

PALETTE = sns.color_palette("tab20", len(KP_NAMES))
KP2COLOR = {kp: PALETTE[i] for i, kp in enumerate(KP_NAMES)}

# ================================================================
# LOAD TXT FILES
# ================================================================
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))
if not TXT_FILES:
    raise RuntimeError("No ABT txt files found.")

# ================================================================
# HELPERS
# ================================================================
def find_processed_mp4(txt_path):
    name = os.path.basename(txt_path)
    cam = re.findall(r"__(\d{3})_", name)[0]
    pattern = os.path.join(os.path.dirname(txt_path), f"July_2025__{cam}_*_2D_result.mp4")
    return glob(pattern)[0]

def get_frame_count(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

# ================================================================
# PARSE FILES
# ================================================================
temp = []
cam_video = {}

for path in TXT_FILES:
    cam_id = re.findall(r"__(\d{3})_", os.path.basename(path))[0]
    mp4_path = find_processed_mp4(path)
    total_frames = get_frame_count(mp4_path)
    cam_video[cam_id] = (mp4_path, total_frames)

    with open(path) as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())
            if not parts or not parts[0].isdigit():
                continue

            frame = int(parts[0])
            kp_vals = parts[7:]
            K = len(kp_vals) // 3

            kpx, kpy = [], []
            for i in range(0, len(kp_vals), 3):
                try:
                    kpx.append(float(kp_vals[i]))
                    kpy.append(float(kp_vals[i+1]))
                except:
                    kpx.append(np.nan)
                    kpy.append(np.nan)

            temp.append([cam_id, frame, total_frames, K, kpx, kpy])

temp_df = pd.DataFrame(temp, columns=["camera","frame","total","K","kpx","kpy"])

# ================================================================
# BUILD FULL TIMELINES
# ================================================================
records = []

for cam_id, sub in temp_df.groupby("camera"):
    total_frames = sub["total"].iloc[0]
    K = sub["K"].iloc[0]

    full_x = np.full((total_frames, K), np.nan)
    full_y = np.full((total_frames, K), np.nan)

    for _, row in sub.iterrows():
        if row["frame"] < total_frames:
            full_x[row["frame"]] = row["kpx"]
            full_y[row["frame"]] = row["kpy"]

    for f in range(total_frames):
        records.append([cam_id, f, K, list(full_x[f]), list(full_y[f])])

df = pd.DataFrame(records, columns=["camera","frame","n_kp","kpx","kpy"])

# ================================================================
# COMPUTE JITTER
# ================================================================
rows_time = []

for cam_id, sub in df.groupby("camera"):
    sub = sub.sort_values("frame")
    xmat = np.vstack(sub["kpx"].values)
    ymat = np.vstack(sub["kpy"].values)

    dist = np.sqrt(np.diff(xmat, axis=0)**2 + np.diff(ymat, axis=0)**2)

    for t in range(dist.shape[0]):
        for k in range(dist.shape[1]):
            rows_time.append([cam_id, t, k, dist[t, k]])

df_time = pd.DataFrame(rows_time, columns=["camera","t","keypoint","jitter"])
df_time["kp_name"] = df_time["keypoint"].map(lambda i: KP_NAMES[i])
df_time["jitter_smooth"] = (
    df_time.groupby(["camera","keypoint"])["jitter"]
    .transform(lambda x: x.rolling(7, center=True, min_periods=1).median())
)

cams_sorted = sorted(df_time["camera"].unique())
n_cams = len(cams_sorted)

# ================================================================
# MASTER FIGURE
# ================================================================
fig = plt.figure(figsize=(18, 10))
gs = GridSpec(n_cams, 3, width_ratios=[1.8, 1.5, 1.1], figure=fig)

for r, cid in enumerate(cams_sorted):

    dfc = df_time[df_time["camera"] == cid]

    # -------- LEFT: TIME SERIES --------
    ax_plot = fig.add_subplot(gs[r, 0])
    for kp in KP_NAMES:
        sub = dfc[dfc["kp_name"] == kp]
        ax_plot.plot(sub["t"], sub["jitter_smooth"], lw=0.6, color=KP2COLOR[kp])

    ax_plot.set_ylabel("jitter (px)", fontsize=13)
    ax_plot.tick_params(axis="y", labelsize=11)

    if r != n_cams - 1:
        ax_plot.set_xticks([])
        ax_plot.set_xlabel("")
    else:
        ax_plot.set_xlabel("frame index", fontsize=13)

    # -------- MIDDLE: BOXPLOT --------
    ax_box = fig.add_subplot(gs[r, 1])

    sns.boxplot(
        data=dfc,
        x="jitter",
        y="kp_name",
        orient="h",
        showcaps=True,
        showfliers=False,
        width=0.6,
        ax=ax_box
    )

    for i, artist in enumerate(ax_box.artists):
        artist.set_facecolor(KP2COLOR[KP_NAMES[i]])
        artist.set_alpha(0.8)

    ax_box.yaxis.tick_left()
    ax_box.tick_params(axis="y", labelsize=10)
    ax_box.set_ylabel("")
    ax_box.set_xlim(0, 100)

    if r != n_cams - 1:
        ax_box.set_xticks([])
        ax_box.set_xlabel("")
    else:
        ax_box.set_xlabel("jitter (px)", fontsize=13)

    # -------- RIGHT: IMAGE --------
    ax_img = fig.add_subplot(gs[r, 2])

    per_frame = dfc.groupby("t")["jitter_smooth"].mean()
    bf = int(per_frame.idxmax()) if not per_frame.dropna().empty else 0

    vid, total = cam_video[cid]
    cap = cv2.VideoCapture(vid)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(bf, total-1))
    ok, frame = cap.read()
    cap.release()

    if ok:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # draw frame index inside image
    cv2.rectangle(frame, (10, 10), (140, 45), (0, 0, 0), -1)
    cv2.putText(frame, f"frame {bf}", (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    ax_img.imshow(frame)
    ax_img.axis("off")

plt.tight_layout(pad=0.15, w_pad=0.15, h_pad=0.15)
plt.show()

if save_plots:
    out = os.path.join(plot_dir, "jitter_full_master_boxplots.png")
    fig.savefig(out, dpi=200)
    print("Saved:", out)

print("\nDone.\n")