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

# Monkey colors
MONKEY_COLORS = {
    "Elm": "#556B2F",   # moss green
    "Jok": "#D96C00"    # dark orange
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
# PARSE TXT INTO (frame, monkey, n_boxes)
# ================================================================
rows = []
for path in TXT_FILES:
    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue
            if not parts[0].isdigit():
                continue
            frame = int(parts[0])
            monkey = parts[1]
            values = parts[2:]
            n = len(values) // 6
            rows.append([frame, monkey, n])

df = pd.DataFrame(rows, columns=["frame", "monkey", "count"])
if df.empty:
    raise RuntimeError("Parsed dataframe is empty — check TXT formatting.")
print("\nParsed dataframe:\n", df.head())

# ================================================================
# PER-FRAME PRESENCE: how many cameras detect each monkey
# ================================================================
df_presence = (
    df.assign(present=(df["count"] > 0).astype(int))
      .groupby(["frame", "monkey"])["present"]
      .sum()
      .reset_index()
)
# present ∈ {0,1,2,3,4}

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
# FRAME → TRUE TIME
# ================================================================
df_presence["time"] = df_presence["frame"].apply(
    lambda f: session_start + timedelta(seconds=f / FPS)
)

# ================================================================
# PIVOT: monkey × time → 0–4 detections
# ================================================================
pivot = df_presence.pivot_table(
    index="monkey",
    columns="time",
    values="present",
    aggfunc="max",
    fill_value=0
)

pivot = pivot.loc[sorted(pivot.index)]   # ensure Elm, Jok order

monkeys = list(pivot.index)
times = list(pivot.columns)
seconds = np.array([(t - session_start).total_seconds() for t in times])
total_sec = seconds.max()

# ================================================================
# COLORMAPS
# ================================================================
# Per-monkey gradient: white → monkey color
row_cmaps = {
    m: LinearSegmentedColormap.from_list(f"cm_{m}", ["#FFFFFF", MONKEY_COLORS[m]], N=5)
    for m in monkeys
}

# Grayscale for legend
legend_cmap = LinearSegmentedColormap.from_list(
    "legend_gray",
    ["#FFFFFF", "#C0C0C0", "#888888", "#555555", "#000000"],
    N=5
)
bounds = [0,1,2,3,4,5]
norm = BoundaryNorm(bounds, legend_cmap.N)

# ================================================================
# BAR PLOT STATS
# ================================================================
stats_rows = []
total_frames = df_presence["frame"].nunique()

for m in monkeys:
    sub = df_presence[df_presence["monkey"] == m]
    for k in [1,2,3,4]:
        abs_count = (sub["present"] == k).sum()
        rel_count = abs_count / total_frames * 100
        stats_rows.append([m, k, abs_count, rel_count])

stats_df = pd.DataFrame(stats_rows, columns=["monkey", "views", "abs", "rel"])
stats_df["views"] = stats_df["views"].astype(int)
stats_df = stats_df.sort_values(["views", "monkey"])

# ================================================================
# FIGURE 1 — HEATMAP
# ================================================================
fig1 = plt.figure(figsize=(8.27, 3.9))
ax1 = fig1.add_subplot(111)

# Draw rows with imshow
for i, monkey in enumerate(monkeys):
    row_vals = pivot.loc[monkey].values.reshape(1, -1)
    ax1.imshow(
        row_vals,
        extent=(0, total_sec, i, i+1),
        cmap=row_cmaps[monkey],
        norm=norm,
        aspect="auto",
        interpolation="nearest"
    )

# ---- FIX 1: equal row height ----
ax1.set_ylim(0, len(monkeys))

# Y labels
ax1.set_yticks(np.arange(len(monkeys)) + 0.5)
ax1.set_yticklabels(monkeys)

# Time axis (elapsed)
major = 10*60
major_ticks = np.arange(0, total_sec+major, major)
if major_ticks[-1] < total_sec:
    major_ticks = np.append(major_ticks, total_sec)

ax1.set_xlim(0, total_sec)
ax1.xaxis.set_major_locator(MultipleLocator(major))
ax1.set_xticks(major_ticks)
ax1.set_xticklabels([f"{int(t//60):02d}:{int(t%60):02d}" for t in major_ticks])

ax1.set_xlabel(
    f"Elapsed time from session start (Start: {session_start.strftime('%H:%M')} — End: {(session_start + timedelta(seconds=total_sec)).strftime('%H:%M')})"
)

ax1.set_title(f"Session {SESSION_DATE} — Presence across 4 cameras")

# Grayscale legend
sm = ScalarMappable(norm=norm, cmap=legend_cmap)
sm.set_array([])
cbar = fig1.colorbar(sm, ax=ax1, pad=0.01, fraction=0.03)
cbar.set_label("Number of cameras detecting monkey")
cbar.set_ticks([0.5,1.5,2.5,3.5,4.5])
cbar.set_ticklabels(["0","1","2","3","4"])

plt.tight_layout()
plt.show()

if save_plots:
    fig1.savefig(os.path.join(plot_dir, f"heatmap_{SESSION_DATE}.png"), dpi=200)

# ================================================================
# FIGURE 2 — BAR PLOT (percent on y, counts as labels)
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

ax2.set_title("Frames detected in 1–4 cameras")
ax2.set_xlabel("Number of cameras detecting each monkey")
ax2.set_ylabel("Relative count (%)")

# ---- FIX 2: correctly centered annotation ----
for patch, (_, row) in zip(ax2.patches, stats_df.iterrows()):
    x = patch.get_x() + patch.get_width() / 2
    y = patch.get_height()
    ax2.text(
        x,
        y + 0.5,
        f"{row['abs']}",
        ha="center",
        va="bottom",
        fontsize=8
    )

ax2.legend(title="Monkey")

plt.tight_layout()
plt.show()

if save_plots:
    fig2.savefig(os.path.join(plot_dir, f"barplot_{SESSION_DATE}.png"), dpi=200)

print("\nDone.\n")