#!/usr/bin/env python3
"""
HomeCagePaper · Section 3 — Behavioral tracking

Loads curated/master.csv, generates publication-ready PNGs in
figures/behavioral/. Uses RT-predicted 3D positions only.

Functional, top-to-bottom in publication order. Re-runs overwrite.

Run:
    python3 HomeCagePaper/curate_data.py     # once
    python3 HomeCagePaper/behavioral_tracking.py
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as sps

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import (                                        # noqa: E402
    apply_style,
    W_SINGLE, W_DOUBLE, PAL_WONG,
    ANIMALS, KEPT_KP_IDX, MM_TO_CM,
)

CURATED = HERE / "curated" / "master.h5"
OUT_DIR = HERE / "_figures" / "behavioral"
TBL_DIR = HERE / "tables"  / "behavioral"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

VIDEO_FPS    = 25.0       # ABT capture rate
MOVE_THRESH  = 5.0        # cm/s — above this is "moving"
MAX_GAP_FR   = 10         # gap larger than this breaks a bout


# ─── 1. Load and reduce to per-frame centroids ───────────────────────────────
def load_centroids():
    """Per (session, animal, frame) trunk centroid (cm) — pre-computed by
    curate_data.py and stored in master.h5 under /centroids."""
    df = pd.read_hdf(CURATED, "centroids")
    df = df.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})
    return df[["session", "animal", "frame", "x", "y", "z"]].sort_values(
        ["session", "animal", "frame"]).reset_index(drop=True)


# ─── 2. Helpers ──────────────────────────────────────────────────────────────
def remove_outliers(df, value_col, group_cols, k=1.5):
    if df.empty:
        return df
    keep = pd.Series(False, index=df.index)
    for _, g in df.groupby(group_cols):
        q1 = g[value_col].quantile(0.25)
        q3 = g[value_col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        keep.loc[g.index] = (g[value_col] >= lo) & (g[value_col] <= hi)
    return df.loc[keep]


def add_speed(centroids, max_gap=MAX_GAP_FR):
    out = []
    for (sess, animal), sub in centroids.groupby(["session", "animal"]):
        s = sub.copy().sort_values("frame").reset_index(drop=True)
        df = s[["x", "y", "z"]].to_numpy()
        fr = s["frame"].to_numpy()
        speeds = np.full(len(s), np.nan)
        for i in range(1, len(s)):
            gap = fr[i] - fr[i-1]
            if gap > max_gap or gap <= 0: continue
            dt = gap / VIDEO_FPS
            speeds[i] = float(np.linalg.norm(df[i] - df[i-1])) / dt
        s["speed_cm_s"] = speeds
        out.append(s)
    return pd.concat(out, ignore_index=True) if out else centroids.copy()


def state_bouts(speed_df, threshold=MOVE_THRESH, max_gap=MAX_GAP_FR):
    rows = []
    for (sess, animal), sub in speed_df.dropna(subset=["speed_cm_s"]).groupby(
            ["session", "animal"]):
        s = sub.sort_values("frame")
        fr = s["frame"].to_numpy()
        sp = s["speed_cm_s"].to_numpy()
        state = sp > threshold
        cur_state = cur_start = cur_last = None
        for i in range(len(fr)):
            st = bool(state[i])
            if cur_state is None:
                cur_state, cur_start, cur_last = st, fr[i], fr[i]; continue
            gap = fr[i] - cur_last
            if st == cur_state and gap <= max_gap:
                cur_last = fr[i]; continue
            rows.append({"session": sess, "animal": animal,
                         "kind": "moving" if cur_state else "still",
                         "duration_s": (cur_last - cur_start) / VIDEO_FPS})
            cur_state, cur_start, cur_last = st, fr[i], fr[i]
        if cur_state is not None:
            rows.append({"session": sess, "animal": animal,
                         "kind": "moving" if cur_state else "still",
                         "duration_s": (cur_last - cur_start) / VIDEO_FPS})
    return pd.DataFrame(rows)


def inter_animal_distance(centroids):
    rows = []
    for sess, sub in centroids.groupby("session"):
        animals = {a: g.set_index("frame") for a, g in sub.groupby("animal")}
        if "Elm" not in animals or "Jok" not in animals: continue
        common = animals["Elm"].index.intersection(animals["Jok"].index)
        if common.empty: continue
        e = animals["Elm"].loc[common, ["x", "y", "z"]].to_numpy()
        j = animals["Jok"].loc[common, ["x", "y", "z"]].to_numpy()
        d = np.linalg.norm(e - j, axis=1)
        for f, dist in zip(common, d):
            rows.append({"session": sess, "frame": int(f), "dist_cm": float(dist)})
    return pd.DataFrame(rows)


# ─── 3. Save helper ──────────────────────────────────────────────────────────
def _save(fig, stem):
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=600)
    print(f"  {stem}.png")
    plt.close(fig)


# ─── 4. Panel drawers (minimalist) ───────────────────────────────────────────
def draw_speed_hist(ax, speed_df):
    df = speed_df.dropna(subset=["speed_cm_s"]).copy()
    if df.empty: return
    df = remove_outliers(df, "speed_cm_s", ["animal"])
    sns.histplot(data=df, x="speed_cm_s", hue="animal",
                 palette=PAL_WONG, bins=25, stat="count",
                 element="step", fill=True, alpha=0.45,
                 common_norm=False, ax=ax)
    ax.axvline(MOVE_THRESH, color="#444", lw=0.6, ls="--",
               label=f">{MOVE_THRESH:.0f} cm/s = moving")
    ax.set_xlabel("speed (cm/s)")
    ax.set_ylabel("count")
    ax.set_title("Speed distribution")
    sns.despine(ax=ax)


def draw_bouts_hist(ax, bouts_df, kind):
    df = bouts_df[bouts_df["kind"] == kind]
    if df.empty: return
    sns.histplot(data=df, x="duration_s", hue="animal",
                 palette=PAL_WONG, bins=20, stat="count",
                 element="step", fill=True, alpha=0.45,
                 common_norm=False, ax=ax)
    ax.set_xlabel("duration (s)")
    ax.set_ylabel("count")
    ax.set_title(f"{kind.capitalize()}-bout durations")
    sns.despine(ax=ax)


def draw_distance(ax, dist_df):
    if dist_df.empty: return
    sns.violinplot(data=dist_df, x="session", y="dist_cm",
                   inner=None, cut=0, color="#cce0f0", ax=ax,
                   linewidth=0.6)
    sns.stripplot(data=dist_df, x="session", y="dist_cm",
                  color="#234", size=2.5, alpha=0.75, jitter=0.18, ax=ax)
    ax.set_ylabel("distance (cm)")
    ax.set_xlabel("session")
    ax.set_title("Inter-animal distance per session")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha("right")
    sns.despine(ax=ax)


# ─── 5. Statistics ───────────────────────────────────────────────────────────
def write_stats(speed_df, bouts_df, dist_df, centroids):
    rows = []

    def mw(label, a, b):
        a = np.asarray(a); b = np.asarray(b)
        if a.size < 5 or b.size < 5:
            rows.append({"test": label, "n_Elm": int(a.size), "n_Jok": int(b.size),
                         "median_Elm": np.nan, "median_Jok": np.nan,
                         "U": np.nan, "p": np.nan}); return
        u, p = sps.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({"test": label, "n_Elm": int(a.size), "n_Jok": int(b.size),
                     "median_Elm": float(np.median(a)),
                     "median_Jok": float(np.median(b)),
                     "U": float(u), "p": float(p)})

    sp = speed_df.dropna(subset=["speed_cm_s"])
    mw("speed (cm/s)",
       sp.loc[sp["animal"] == "Elm", "speed_cm_s"],
       sp.loc[sp["animal"] == "Jok", "speed_cm_s"])
    mw("height z (cm)",
       centroids.loc[centroids["animal"] == "Elm", "z"],
       centroids.loc[centroids["animal"] == "Jok", "z"])
    mov = bouts_df[bouts_df["kind"] == "moving"]
    mw("movement bout (s)",
       mov.loc[mov["animal"] == "Elm", "duration_s"],
       mov.loc[mov["animal"] == "Jok", "duration_s"])
    stl = bouts_df[bouts_df["kind"] == "still"]
    mw("still bout (s)",
       stl.loc[stl["animal"] == "Elm", "duration_s"],
       stl.loc[stl["animal"] == "Jok", "duration_s"])

    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "stats.csv", index=False)
    print("\nMann-Whitney U (two-sided):")
    print(out.to_string(index=False))


# ─── 6. Figure builders (chronological) ──────────────────────────────────────
def build_figure1(speed_df, bouts_df):
    """Locomotion — speed distribution and bout durations."""
    print("\n── figure1 — Locomotion ──")

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, W_DOUBLE * 0.34))
    draw_speed_hist(axes[0], speed_df)
    draw_bouts_hist(axes[1], bouts_df, "moving")
    draw_bouts_hist(axes[2], bouts_df, "still")
    fig.subplots_adjust(wspace=0.40)
    _save(fig, "figure1")

    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_speed_hist(ax, speed_df); _save(fig, "figure1_a")

    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_bouts_hist(ax, bouts_df, "moving")
    _save(fig, "figure1_b")

    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_bouts_hist(ax, bouts_df, "still")
    _save(fig, "figure1_c")


def build_figure2(dist_df):
    """Social — inter-animal distance."""
    print("\n── figure2 — Inter-animal distance ──")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE * 0.72))
    draw_distance(ax, dist_df)
    _save(fig, "figure2")


# ─── 7. main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {CURATED.name}")
    centroids = load_centroids()
    speed_df  = add_speed(centroids)
    bouts_df  = state_bouts(speed_df)
    dist_df   = inter_animal_distance(centroids)
    print(f"  centroids: {len(centroids)}  "
          f"speeds: {speed_df['speed_cm_s'].notna().sum()}  "
          f"bouts: {len(bouts_df)}  "
          f"inter-animal distances: {len(dist_df)}")
    print(f"\nWriting figures to {OUT_DIR}")

    build_figure1(speed_df, bouts_df)
    build_figure2(dist_df)
    write_stats(speed_df, bouts_df, dist_df, centroids)

    print(f"\nDone. {len(list(OUT_DIR.glob('figure*.png')))} PNGs.")


if __name__ == "__main__":
    main()
