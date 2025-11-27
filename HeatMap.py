#!/usr/bin/env python3

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import MultipleLocator, FuncFormatter
from glob import glob

# ================================================================
# CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage"
FPS = 5

plot_dir = os.path.join(BASE, "plots")
os.makedirs(plot_dir, exist_ok=True)

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
# PARSE: frame, monkey_id, count_per_view
# ================================================================
rows = []

"""
TXT format:
frame monkey_id x1 y1 x2 y2 conf class x1 y1 x2 y2 conf class ...
"""

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

            rows.append([frame_idx, monkey_id, n_boxes])

df = pd.DataFrame(rows, columns=["frame", "monkey", "count"])

if df.empty:
    raise RuntimeError("Parsed dataframe is empty — check TXT formatting.")

print("\nParsed dataframe:\n", df.head())

# ================================================================
# AGGREGATE ACROSS 4 CAMERAS
# ================================================================
df_view_presence = (
    df.assign(present=(df["count"] > 0).astype(int))
      .groupby(["frame", "monkey"])["present"]
      .sum()
      .reset_index()
)

# Now presence ∈ {0,1,2,3,4}

# ================================================================
# EXTRACT START TIME FROM FILENAME
# ================================================================
first_name = os.path.basename(TXT_FILES[0])
timestamps = re.findall(r"(\d{14})", first_name)

if len(timestamps) < 1:
    raise RuntimeError("Could not extract recording timestamp.")

recording_timestamp = timestamps[0]   # FIRST timestamp = recording start
session_start = datetime.strptime(recording_timestamp, "%Y%m%d%H%M%S")
SESSION_DATE = recording_timestamp[:8]

print("\nRecording start =", session_start)

# ================================================================
# FRAME → TRUE TIME
# ================================================================
df_view_presence["time"] = df_view_presence["frame"].apply(
    lambda f: session_start + timedelta(seconds=f / FPS)
)

# ================================================================
# PIVOT: monkey × time (presence 0–4)
# ================================================================
pivot_presence = df_view_presence.pivot_table(
    index="monkey",
    columns="time",
    values="present",
    aggfunc="max",
    fill_value=0
)

# ================================================================
# GREEN GRADIENT (0–4 views)
# ================================================================
colors = ["#FFFFFF", "#DDE8D3", "#AFC89A", "#7E995F", "#556B2F"]
cmap = LinearSegmentedColormap.from_list("presence_green", colors, N=5)

bounds = [0, 1, 2, 3, 4, 5]   # 0–4 inclusive
norm = BoundaryNorm(bounds, cmap.N)

# ================================================================
# PLOT HEATMAP
# ================================================================
plt.figure(figsize=(20, 4))

ax = sns.heatmap(
    pivot_presence,
    cmap=cmap,
    norm=norm,
    cbar=True,
    linewidths=0,
    xticklabels=False,
    cbar_kws={
        "ticks": [0.5, 1.5, 2.5, 3.5, 4.5],
        "format": lambda val, _: int(val - 0.5),
        "label": "Presence across 4 views (0–4)"
    }
)

plt.title(f"Monkey Presence (0–4 Cameras) — {SESSION_DATE}", fontsize=18)
plt.ylabel("Monkey", fontsize=14)

times = list(pivot_presence.columns)
seconds = np.array([(t - session_start).total_seconds() for t in times])

ax.set_xlim(0, seconds.max())

# Determine recording duration
total_sec = seconds.max()

# Major ticks every 10 minutes
major_step = 10 * 60
major_ticks = np.arange(0, total_sec + major_step, major_step)

# Guarantee last tick exactly at end
if major_ticks[-1] < total_sec:
    major_ticks = np.append(major_ticks, total_sec)

# Minor ticks every 2 minutes
minor_step = 2 * 60
minor_ticks = np.arange(0, total_sec + minor_step, minor_step)

ax.xaxis.set_major_locator(MultipleLocator(major_step))
ax.xaxis.set_minor_locator(MultipleLocator(minor_step))

# Formatter for HH:MM
def time_formatter(x, pos):
    t = session_start + timedelta(seconds=float(x))
    return t.strftime("%H:%M")

ax.set_xticks(major_ticks)
ax.set_xticklabels([time_formatter(t, None) for t in major_ticks])

# No angled labels
plt.xticks(rotation=0)