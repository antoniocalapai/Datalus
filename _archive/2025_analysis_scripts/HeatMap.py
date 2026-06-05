#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator
from matplotlib.cm import ScalarMappable
from glob import glob

# ================================================================
# GLOBAL CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage"
FPS = 5

save_plots = True
plot_dir = os.path.join(BASE, "plots")
os.makedirs(plot_dir, exist_ok=True)

MONKEY_COLORS = {
    "Elm": "#556B2F",
    "Jok": "#D96C00"
}

sns.set_theme(style="whitegrid")

# ================================================================
# LOAD TXT FILES
# ================================================================
TXT_FILES = sorted(glob(os.path.join(BASE, "*_2D_result.txt")))
if len(TXT_FILES) != 4:
    raise RuntimeError(f"Expected 4 TXT files, found {len(TXT_FILES)}")

print("\nUsing TXT files:")
for t in TXT_FILES:
    print(" -", t)

# ================================================================
# EXTRACT SESSION START TIME
# ================================================================
first_name = os.path.basename(TXT_FILES[0])
timestamps = re.findall(r"(\d{14})", first_name)
if not timestamps:
    raise RuntimeError("Could not extract timestamp from filename.")

recording_timestamp = timestamps[0]
session_start = datetime.strptime(recording_timestamp, "%Y%m%d%H%M%S")
SESSION_DATE = recording_timestamp[:8]

print("\nRecording start =", session_start)

# ================================================================
# TRUE MULTIVIEW PARSING
# ================================================================
def extract_cam_id(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else "???"

rows = []

for path in TXT_FILES:
    cam_id = extract_cam_id(path)

    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            if not parts[0].isdigit():
                continue

            frame_local = int(parts[0])
            monkey = parts[1]
            values = parts[2:]
            n_boxes = len(values) // 6

            present = 1 if n_boxes > 0 else 0

            # local time → shared multiview tick
            t_sec = frame_local / FPS
            tick = int(round(t_sec * FPS))

            rows.append([tick, cam_id, monkey, present])

df_mv = pd.DataFrame(rows, columns=["tick", "cam", "monkey", "present"])
if df_mv.empty:
    raise RuntimeError("Parsed multiview dataframe is empty.")

# per (tick, monkey, cam)
df_mv = (
    df_mv.groupby(["tick", "monkey", "cam"], as_index=False)["present"]
         .max()
)

# per (tick, monkey): number of cameras detecting the monkey
df_presence = (
    df_mv.groupby(["tick", "monkey"], as_index=False)["present"]
         .sum()
         .rename(columns={"present": "n_cams"})
)

# sanity
assert df_presence["n_cams"].between(0, 4).all()

df_presence["time"] = df_presence["tick"].apply(
    lambda k: session_start + timedelta(seconds=k / FPS)
)

# ================================================================
# HEATMAP DATA
# ================================================================
pivot = df_presence.pivot_table(
    index="monkey",
    columns="time",
    values="n_cams",
    aggfunc="max",
    fill_value=0
)

pivot = pivot.loc[sorted(pivot.index)]
monkeys = list(pivot.index)

times = list(pivot.columns)
seconds = np.array([(t - session_start).total_seconds() for t in times])
total_sec = seconds.max() if len(seconds) else 0.0

row_cmaps = {
    m: LinearSegmentedColormap.from_list(
        f"cm_{m}", ["#FFFFFF", MONKEY_COLORS[m]], N=5
    )
    for m in monkeys
}

legend_cmap = LinearSegmentedColormap.from_list(
    "legend_gray",
    ["#FFFFFF", "#C0C0C0", "#888888", "#555555", "#000000"],
    N=5
)
norm = BoundaryNorm([0, 1, 2, 3, 4, 5], legend_cmap.N)

# ================================================================
# FIGURE 1 — HEATMAP
# ================================================================
fig1 = plt.figure(figsize=(8.27, 3.9))
ax1 = fig1.add_subplot(111)

for i, monkey in enumerate(monkeys):
    row_vals = pivot.loc[monkey].values.reshape(1, -1)
    ax1.imshow(
        row_vals,
        extent=(0, total_sec, i, i + 1),
        cmap=row_cmaps[monkey],
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

ax1.set_ylim(0, len(monkeys))
ax1.set_yticks(np.arange(len(monkeys)) + 0.5)
ax1.set_yticklabels(monkeys)

major = 10 * 60
ax1.set_xlim(0, total_sec)
ax1.xaxis.set_major_locator(MultipleLocator(major))

ax1.set_xticks(np.arange(0, total_sec + major, major))
ax1.set_xticklabels(
    [f"{int(t//60):02d}:{int(t%60):02d}"
     for t in np.arange(0, total_sec + major, major)]
)

ax1.set_xlabel(
    f"Elapsed time from session start (Start: {session_start.strftime('%H:%M')})"
)
ax1.set_title(f"Session {SESSION_DATE} — Multiview presence (4 cameras)")

sm = ScalarMappable(norm=norm, cmap=legend_cmap)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1, pad=0.01, fraction=0.03)
cbar.set_label("Number of cameras detecting monkey")
cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
cbar.set_ticklabels(["0", "1", "2", "3", "4"])

plt.tight_layout()
plt.show()

if save_plots:
    fig1.savefig(os.path.join(plot_dir, f"heatmap_{SESSION_DATE}.png"), dpi=200)

# ================================================================
# BAR STATS — OPTION 2 (CORRECT)
# ================================================================
stats = []

for m in monkeys:
    sub = df_presence[df_presence["monkey"] == m]
    detected = sub[sub["n_cams"] >= 1]

    denom = len(detected)
    for k in [1, 2, 3, 4]:
        abs_count = int((detected["n_cams"] == k).sum())
        rel = abs_count / denom * 100 if denom > 0 else 0.0
        stats.append([m, k, abs_count, rel, denom])

stats_df = pd.DataFrame(
    stats, columns=["monkey", "views", "abs", "rel", "denom"]
)

# ================================================================
# FIGURE 2 — BAR PLOT
# ================================================================
fig2 = plt.figure(figsize=(8.27, 3.9))
ax2 = fig2.add_subplot(111)

sns.barplot(
    data=stats_df,
    x="views",
    y="rel",
    hue="monkey",
    palette=MONKEY_COLORS,
    ax=ax2
)

ax2.set_title(
    "Distribution of multi-view detections"
)
ax2.set_xlabel("Number of cameras")
ax2.set_ylabel("% of multi-view frames (>0 detections)")

for patch, (_, row) in zip(ax2.patches, stats_df.iterrows()):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    ax2.text(x, y + 0.8, f"{row['abs']}", ha="center", va="bottom", fontsize=8)


ax2.legend(title="Monkey")
plt.tight_layout()
plt.show()

if save_plots:
    fig2.savefig(os.path.join(plot_dir, f"barplot_{SESSION_DATE}.png"), dpi=200)

print("\nDone — multiview statistics are now consistent.\n")