#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import cv2
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

# ================================================================
# GLOBAL CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage"
FPS = 5

# User requested: only display (but keep a switch)
save_plots = True

plot_dir = os.path.join(BASE, "plots")
os.makedirs(plot_dir, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

mpl.rcParams.update({
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
})

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

# Global color palette for keypoints
PALETTE = sns.color_palette("tab20", len(KP_NAMES))
KP2COLOR = {kp: PALETTE[i] for i, kp in enumerate(KP_NAMES)}

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
def find_processed_mp4(txt_path: str) -> str:
    name = os.path.basename(txt_path)
    m = re.search(r"__(\d{3})_", name)
    if not m:
        raise RuntimeError(f"Could not parse camera id from: {name}")
    cam = m.group(1)
    pattern = os.path.join(os.path.dirname(txt_path), f"July_2025__{cam}_*_2D_result.mp4")
    files = glob(pattern)
    if not files:
        raise RuntimeError(f"No processed MP4 found for camera {cam} using pattern: {pattern}")
    return files[0]

def get_frame_count(mp4_path: str) -> int:
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

def draw_frame_index_on_image(rgb_img: np.ndarray, frame_idx: int) -> np.ndarray:
    """Overlay frame number inside the image (top-left) with dark background."""
    img = rgb_img.copy()
    text = f"frame {frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 12, 28
    cv2.rectangle(img, (x - 6, y - th - 10), (x + tw + 6, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img

# ================================================================
# PARSE KEYPOINT FILES
# ================================================================
temp = []
cam_video = {}

for path in TXT_FILES:
    cam_id = re.findall(r"__(\d{3})_", os.path.basename(path))[0]
    mp4_path = find_processed_mp4(path)
    total_frames = get_frame_count(mp4_path)
    cam_video[cam_id] = (mp4_path, total_frames)

    with open(path, "r") as f:
        for line in f:
            parts = re.split(r"[,\s]+", line.strip())
            if not parts or parts[0] == "frame_number":
                continue

            try:
                frame = int(parts[0])
            except Exception:
                continue

            kp_vals = parts[7:]
            if not kp_vals:
                continue

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

temp_df = pd.DataFrame(temp, columns=["camera", "frame", "total", "K", "kpx", "kpy"])
if temp_df.empty:
    raise RuntimeError("Parsed temp_df is empty — check TXT formatting or file content.")

# ================================================================
# BUILD FULL TIMELINES (NAN-FILLED)
# ================================================================
records = []
for cam_id, sub in temp_df.groupby("camera"):
    total_frames = int(sub["total"].iloc[0])
    K = int(sub["K"].iloc[0])

    full_x = np.full((total_frames, K), np.nan, dtype=np.float32)
    full_y = np.full((total_frames, K), np.nan, dtype=np.float32)

    for _, row in sub.iterrows():
        f = int(row["frame"])
        if 0 <= f < total_frames:
            full_x[f] = np.array(row["kpx"], dtype=np.float32)
            full_y[f] = np.array(row["kpy"], dtype=np.float32)

    for f in range(total_frames):
        records.append([cam_id, f, K, list(full_x[f]), list(full_y[f])])

df = pd.DataFrame(records, columns=["camera", "frame", "n_kp", "kpx", "kpy"])

# ================================================================
# COMPUTE JITTER
# ================================================================
jitter_time_rows = []
summary_rows = []

for cam_id, subdf in df.groupby("camera"):
    subdf = subdf.sort_values("frame")
    K = int(subdf.iloc[0]["n_kp"])

    xmat = np.vstack(subdf["kpx"].values)
    ymat = np.vstack(subdf["kpy"].values)

    dx = xmat[1:] - xmat[:-1]
    dy = ymat[1:] - ymat[:-1]
    dist = np.sqrt(dx**2 + dy**2)  # [T-1, K]

    for k in range(K):
        summary_rows.append([cam_id, k, float(np.nanmean(dist[:, k]))])

    T = dist.shape[0]
    for t in range(T):
        for k in range(K):
            jitter_time_rows.append([cam_id, t, k, float(dist[t, k])])

jdf = pd.DataFrame(summary_rows, columns=["camera", "keypoint", "mean_jitter"])
jdf["kp_name"] = jdf["keypoint"].apply(lambda i: KP_NAMES[int(i)])

df_time = pd.DataFrame(jitter_time_rows, columns=["camera", "t", "keypoint", "jitter"])
df_time["kp_name"] = df_time["keypoint"].apply(lambda i: KP_NAMES[int(i)])

# smoothing
df_time["jitter_smooth"] = (
    df_time.groupby(["camera", "keypoint"])["jitter"]
    .transform(lambda x: x.rolling(7, center=True, min_periods=1).median())
)

cams_sorted = sorted(df_time["camera"].unique())
n_cams = len(cams_sorted)

# ================================================================
# GLOBAL Y LIMIT (for consistent aesthetics)
# ================================================================
vals = df_time["jitter_smooth"].to_numpy()
vals = vals[np.isfinite(vals)]
if vals.size:
    GLOBAL_YMAX = float(np.nanpercentile(vals, 99.5))
    GLOBAL_YMAX = max(50.0, min(GLOBAL_YMAX, 2000.0))
else:
    GLOBAL_YMAX = 100.0

# ================================================================
# MASTER FIGURE
# ================================================================
fig = plt.figure(figsize=(18, 10), constrained_layout=True)
gs = GridSpec(n_cams, 3, width_ratios=[2.0, 1.6, 1.0], figure=fig)

ax_plot_ref = None

for row_idx, cid in enumerate(cams_sorted):
    dfc = df_time[df_time["camera"] == cid].copy()

    # ---------------- LEFT: TIME PLOT ----------------
    if ax_plot_ref is None:
        ax_plot = fig.add_subplot(gs[row_idx, 0])
        ax_plot_ref = ax_plot
    else:
        ax_plot = fig.add_subplot(gs[row_idx, 0], sharex=ax_plot_ref)

    for kp in KP_NAMES:
        sub = dfc[dfc["kp_name"] == kp]
        ax_plot.plot(sub["t"], sub["jitter_smooth"], linewidth=0.7, color=KP2COLOR[kp])

    # aesthetics: no titles, tighter, remove x-axis for non-bottom
    ax_plot.set_title("")
    ax_plot.set_ylabel("jitter (px)")
    ax_plot.set_ylim(0, 2000)
    ax_plot.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_plot.grid(True, axis="y", alpha=0.25)

    if row_idx != n_cams - 1:
        ax_plot.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_plot.set_xlabel("")
    else:
        ax_plot.set_xlabel("frame index")
        ax_plot.tick_params(axis="x", labelsize=10)

    ax_plot.tick_params(axis="y", labelsize=10)

    # ---------------- MIDDLE: BOXPLOT ----------------
    ax_box = fig.add_subplot(gs[row_idx, 1])

    sns.boxplot(
        data=dfc,
        x="jitter",
        y="kp_name",
        orient="h",
        showcaps=True,
        showfliers=False,
        width=0.7,
        ax=ax_box
    )

    # Recolor boxes (robustly: seaborn stores artists per category)
    # Note: box patches are in ax_box.artists for this seaborn version
    for i, artist in enumerate(ax_box.artists):
        if i < len(KP_NAMES):
            kp = KP_NAMES[i]
            artist.set_facecolor(KP2COLOR[kp])
            artist.set_edgecolor("black")
            artist.set_alpha(0.8)

    # recolor whiskers/caps/medians (lines appear in groups; keep it simple: black w/alpha)
    for ln in ax_box.lines:
        ln.set_alpha(0.75)

    # labels on the LEFT + smaller + closer
    ax_box.yaxis.tick_left()
    ax_box.yaxis.set_label_position("left")
    ax_box.tick_params(axis="y", labelsize=9, pad=2)

    # remove titles, tighten, remove x-axis on non-bottom
    ax_box.set_title("")
    ax_box.set_xlim(0, 100)
    ax_box.grid(True, axis="x", alpha=0.25)

    if row_idx != n_cams - 1:
        ax_box.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_box.set_xlabel("")
    else:
        ax_box.set_xlabel("jitter (px)")
        ax_box.tick_params(axis="x", labelsize=10)

    # ---------------- RIGHT: REPRESENTATIVE FRAME ----------------
    ax_img = fig.add_subplot(gs[row_idx, 2])

    per_frame_j = dfc.groupby("t")["jitter_smooth"].mean()
    per_frame_j = per_frame_j.replace([np.inf, -np.inf], np.nan)

    if per_frame_j.dropna().empty:
        best_frame = 0
    else:
        best_frame = int(per_frame_j.idxmax())

    vid_path, total_frames = cam_video[cid]
    bf = max(0, min(best_frame, int(total_frames) - 1))

    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, bf)
    ok, frame = cap.read()
    cap.release()

    if ok and frame is not None:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = draw_frame_index_on_image(frame_rgb, bf)
    else:
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_rgb = draw_frame_index_on_image(frame_rgb, bf)

    ax_img.imshow(frame_rgb)
    ax_img.set_title("")  # no titles
    ax_img.set_aspect("auto")
    ax_img.axis("off")
    ax_img.margins(0)

# show only (no saving)
plt.show()

if save_plots:
    out_path = os.path.join(plot_dir, "jitter_full_master_boxplots.png")
    fig.savefig(out_path, dpi=200)
    print("Saved figure:", out_path)

print("\nDone.\n")