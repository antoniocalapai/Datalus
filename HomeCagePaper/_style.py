"""Shared style constants. Tiny, no logic.
   Imported by validation.py / spatial_analysis.py / behavioral_tracking.py.
"""
import matplotlib as mpl
import seaborn as sns

# Page widths (Nature Communications)
MM_PER_INCH = 25.4
W_SINGLE = 88  / MM_PER_INCH    # 3.46 in
W_15COL  = 120 / MM_PER_INCH    # 4.72 in
W_DOUBLE = 180 / MM_PER_INCH    # 7.09 in

# Wong colour-blind-safe palette
PAL_WONG = {
    "Elm":     "#0072B2",
    "Jok":     "#E69F00",
    "accent":  "#009E73",
    "warn":    "#D55E00",
    "neutral": "#999999",
}

ANIMALS  = ["Elm", "Jok"]
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
__all__ = ["W_SINGLE","W_15COL","W_DOUBLE","PAL_WONG","ANIMALS","KP_NAMES",
           "KEPT_KP_IDX","COCO_PAIRS_BODY","ROOM_MM","ROOM_CM","ROOM_DIAG_CM",
           "MM_TO_CM","apply_style","STRIP_KW"]
KEPT_KP_IDX = [5, 6, 11, 12, 13, 14]
COCO_PAIRS_BODY = [
    (5, 6),                  # shoulders
    (5, 11), (6, 12),        # torso sides
    (11, 12),                # hips
    (11, 13), (12, 14),      # upper legs
]

ROOM_MM      = {"x": 2240, "y": 3400, "z": 3260}
MM_TO_CM     = 0.1
ROOM_CM      = {k: v * MM_TO_CM for k, v in ROOM_MM.items()}
ROOM_DIAG_CM = float((sum(v**2 for v in ROOM_MM.values())) ** 0.5) * MM_TO_CM


def apply_style():
    mpl.rcParams.update({
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         7,
        "axes.titlesize":    8,
        "axes.titleweight":  "bold",
        "axes.labelsize":    7,
        "axes.linewidth":    0.5,
        "xtick.labelsize":   6,
        "ytick.labelsize":   6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size":  2,
        "ytick.major.size":  2,
        "legend.fontsize":   6,
        "legend.frameon":    False,
        "lines.linewidth":   0.75,
        "savefig.dpi":       600,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.04,
    })
    sns.set_theme(style="ticks", context="paper", rc=mpl.rcParams)


STRIP_KW = dict(size=1.6, alpha=0.55, jitter=0.18,
                edgecolor="black", linewidth=0.15)
