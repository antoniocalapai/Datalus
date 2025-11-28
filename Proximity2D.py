#!/usr/bin/env python3

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from collections import defaultdict

# ================================================================
# CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage"
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
# PARSE TXT FILES
# Goal: frame → monkey → number of cameras where detected
# ================================================================
# dictionary: frame -> monkey -> count of cameras where detected
det_per_frame = defaultdict(lambda: {"Elm": 0, "Jok": 0})

for path in TXT_FILES:
    with open(path, "r") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                continue

            # frame index
            if not parts[0].isdigit():
                continue

            frame = int(parts[0])
            monkey = parts[1]

            # bounding boxes in this camera
            values = parts[2:]
            n_boxes = len(values) // 6
            if n_boxes > 0:
                det_per_frame[frame][monkey] += 1  # +1 camera detects this monkey

# Convert to DataFrame
rows = []
for frame, d in det_per_frame.items():
    rows.append([frame, d["Elm"], d["Jok"]])

df = pd.DataFrame(rows, columns=["frame", "Elm_views", "Jok_views"])
df = df.sort_values("frame")

print("\nParsed detection counts per frame:\n", df.head())

# ================================================================
# FIND FRAMES WHERE BOTH MONKEYS ARE SEEN BY ≥2 CAMERAS
# ================================================================
df["both_seen_2plus"] = (
    (df["Elm_views"] >= 2) &
    (df["Jok_views"] >= 2)
).astype(int)

absolute_count = df["both_seen_2plus"].sum()
relative_percentage = 100 * absolute_count / len(df)

print("\n=================================================")
print("Frames where BOTH monkeys appear in ≥ 2 cameras")
print("Absolute count :", absolute_count)
print(f"Relative count : {relative_percentage:.2f}%")
print("=================================================\n")

# ================================================================
# PLOTS: ABSOLUTE + RELATIVE
# ================================================================
sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --------------------------
# 1) Absolute count
# --------------------------
axes[0].bar(["Both ≥2 views"], [absolute_count], color="#556B2F")
axes[0].set_title("Absolute Count", fontsize=14)
axes[0].set_ylabel("Number of frames", fontsize=12)

# --------------------------
# 2) Relative %
# --------------------------
axes[1].bar(["Both ≥2 views"], [relative_percentage], color="#7E995F")
axes[1].set_title("Relative Percentage", fontsize=14)
axes[1].set_ylabel("% of session frames", fontsize=12)
axes[1].set_ylim(0, 100)

plt.suptitle("Co-Visibility of Monkeys (≥2 Cameras Each)", fontsize=16)
plt.tight_layout()

out_path = os.path.join(plot_dir, "both_monkeys_2plus_visibility.png")
plt.savefig(out_path, dpi=150)
plt.show()

print(f"\nSaved: {out_path}\n")