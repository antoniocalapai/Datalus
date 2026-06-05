#!/usr/bin/env python3
"""
HomeCagePaper · make_talk_figure.py

Single composite figure for tomorrow's 10-minute talk.
Three panels in a row:
    a   3D keypoint error  (validation)
    b   3D occupancy        (spatial behaviour)
    c   inter-animal distance (social)

Output: figures/talk_figure.png
"""
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import (apply_style, PAL_WONG, W_SINGLE,
                    ANIMALS, KEPT_KP_IDX, KP_NAMES, COCO_PAIRS_BODY,
                    MM_TO_CM, ROOM_CM, STRIP_KW)
apply_style()

CURATED  = HERE / "curated" / "master.h5"
OUT_DIR  = HERE / "_figures"
VOXEL_CM = 25.0


# ─── Load data ───────────────────────────────────────────────────────────────
def load_validation_2d():
    """2D pixel error per matched (cam, frame, animal, kp).
    Returns (df, n_gt_frames) where n_gt_frames is the number of unique
    (session, frame) tuples with at least one matched GT∩RT 2D detection."""
    kp = pd.read_hdf(CURATED, "keypoints")
    gt = kp[kp["model"]=="GT"][["session","cam","frame","animal","kp_idx","x_2d","y_2d"]]
    rt = kp[kp["model"]=="RT"][["session","cam","frame","animal","kp_idx","x_2d","y_2d"]]
    m = gt.merge(rt, on=["session","cam","frame","animal","kp_idx"],
                 suffixes=("_gt","_rt"))
    m = m.dropna(subset=["x_2d_gt","y_2d_gt","x_2d_rt","y_2d_rt"])
    m["err_px"] = np.hypot(m["x_2d_gt"]-m["x_2d_rt"], m["y_2d_gt"]-m["y_2d_rt"])
    n_frames = m[["session","frame"]].drop_duplicates().shape[0]
    return m[["cam","animal","err_px"]], n_frames


def load_validation_3d():
    """3D cm error, one row per matched (frame, animal, kp). Returns
    (df, n_gt_frames). 3D errors do not have per-camera attribution — once
    triangulated, the value lives in the world frame independent of which
    cameras contributed."""
    kp = pd.read_hdf(CURATED, "keypoints")
    de = (kp.dropna(subset=["x_3d"])
            .drop_duplicates(["session","model","frame","animal","kp_idx"]))
    gt = de[de["model"]=="GT"].drop(columns="model")
    rt = de[de["model"]=="RT"].drop(columns="model")
    m = gt.merge(rt, on=["session","frame","animal","kp_idx"],
                 suffixes=("_gt","_rt"))
    dx = (m["x_3d_rt"] - m["x_3d_gt"]) * MM_TO_CM
    dy = (m["y_3d_rt"] - m["y_3d_gt"]) * MM_TO_CM
    dz = (m["z_3d_rt"] - m["z_3d_gt"]) * MM_TO_CM
    m["err_cm"] = np.sqrt(dx**2 + dy**2 + dz**2)
    n_frames = m[["session","frame"]].drop_duplicates().shape[0]
    return m[["animal","err_cm"]], n_frames


def load_centroids():
    df = pd.read_hdf(CURATED, "centroids")
    df = df.rename(columns={"x_cm":"x","y_cm":"y","z_cm":"z"})
    return df[["session","animal","frame","x","y","z"]]


def inter_animal_distance(centroids):
    rows = []
    for sess, sub in centroids.groupby("session"):
        animals = {a: g.set_index("frame") for a, g in sub.groupby("animal")}
        if "Elm" not in animals or "Jok" not in animals: continue
        common = animals["Elm"].index.intersection(animals["Jok"].index)
        if common.empty: continue
        e = animals["Elm"].loc[common, ["x","y","z"]].to_numpy()
        j = animals["Jok"].loc[common, ["x","y","z"]].to_numpy()
        d = np.linalg.norm(e - j, axis=1)
        for f, dist in zip(common, d):
            rows.append({"session": sess, "frame": int(f), "dist_cm": float(dist)})
    return pd.DataFrame(rows)


def remove_outliers(df, col, group_cols, k=1.5):
    if df.empty: return df
    keep = pd.Series(False, index=df.index)
    for _, g in df.groupby(group_cols):
        q1, q3 = g[col].quantile(0.25), g[col].quantile(0.75)
        iqr = q3 - q1
        keep.loc[g.index] = (g[col] >= q1 - k*iqr) & (g[col] <= q3 + k*iqr)
    return df.loc[keep]


# ─── Panel drawers ───────────────────────────────────────────────────────────
def draw_2d_error_per_cam(ax, df_2d, n_gt_frames):
    df = remove_outliers(df_2d, "err_px", ["animal","cam"])
    cams = sorted(df["cam"].unique())
    sns.violinplot(data=df, x="cam", y="err_px", hue="animal",
                   palette=PAL_WONG, split=True, inner="quartile",
                   cut=0, density_norm="width", linewidth=0.6,
                   order=cams, ax=ax, legend=False)
    sns.stripplot(data=df, x="cam", y="err_px", hue="animal",
                  palette=PAL_WONG, dodge=True, order=cams,
                  ax=ax, legend=False, **STRIP_KW)
    ax.set_xticks(range(len(cams)))
    ax.set_xticklabels([f"{c}" for c in cams])
    ax.set_xlabel("camera")
    ax.set_ylabel("2D error (px)")
    ax.set_title(f"2D keypoint error  ({n_gt_frames} GT frames · {len(df)} kp)")
    sns.despine(ax=ax)


def draw_3d_error_per_animal(ax, df_3d, n_gt_frames):
    df = remove_outliers(df_3d, "err_cm", ["animal"])
    sns.violinplot(data=df, x="animal", y="err_cm", hue="animal",
                   palette=PAL_WONG, inner="quartile", cut=0,
                   density_norm="width", linewidth=0.6,
                   order=ANIMALS, legend=False, ax=ax)
    sns.stripplot(data=df, x="animal", y="err_cm", hue="animal",
                  palette=PAL_WONG, order=ANIMALS, ax=ax,
                  legend=False, **STRIP_KW)
    ax.set_xlabel("")
    ax.set_ylabel("3D error (cm)")
    ax.set_title(f"3D keypoint error  ({n_gt_frames} GT frames · {len(df)} kp)")
    sns.despine(ax=ax)


def draw_3d_voxels(ax, centroids):
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]
    corners = [(0,0,0),(rx,0,0),(rx,ry,0),(0,ry,0),
               (0,0,rz),(rx,0,rz),(rx,ry,rz),(0,ry,rz)]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]),
                color="#666", lw=0.6, alpha=0.8)

    nx = int(np.ceil(rx / VOXEL_CM))
    ny = int(np.ceil(ry / VOXEL_CM))
    nz = int(np.ceil(rz / VOXEL_CM))
    x_e = np.arange(nx + 1) * VOXEL_CM
    y_e = np.arange(ny + 1) * VOXEL_CM
    z_e = np.arange(nz + 1) * VOXEL_CM
    X, Y, Z = np.meshgrid(x_e, y_e, z_e, indexing="ij")
    for animal in ANIMALS:
        sub = centroids[centroids["animal"] == animal]
        grid = np.zeros((nx, ny, nz), dtype=int)
        if not sub.empty:
            ix = np.clip((sub["x"] / VOXEL_CM).astype(int), 0, nx - 1)
            iy = np.clip((sub["y"] / VOXEL_CM).astype(int), 0, ny - 1)
            iz = np.clip((sub["z"] / VOXEL_CM).astype(int), 0, nz - 1)
            for a_, b_, c_ in zip(ix, iy, iz):
                grid[a_, b_, c_] += 1
        if not grid.any(): continue
        filled = grid > 0
        max_c = grid.max()
        alpha = 0.20 + 0.65 * (grid / max_c)
        rgba = np.zeros(filled.shape + (4,))
        base = mpl.colors.to_rgb(PAL_WONG.get(animal, "#888"))
        rgba[..., 0] = base[0]; rgba[..., 1] = base[1]; rgba[..., 2] = base[2]
        rgba[..., 3] = np.where(filled, alpha, 0.0)
        ax.voxels(X, Y, Z, filled, facecolors=rgba,
                  edgecolor=(0, 0, 0, 0.20), linewidth=0.2, shade=True)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_xlim(0, rx); ax.set_ylim(0, ry); ax.set_zlim(0, rz)
    ax.set_box_aspect((rx, ry, rz))
    ax.set_title(f"3D occupancy ({VOXEL_CM:.0f} cm voxels)")


_FULL_COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


def canonical_sitting_pose(room_cm):
    """Hand-built sitting macaque, ~55 cm tall, centred on the room floor.
    Returns dict kp_idx → (x, y, z) in cm."""
    cx = room_cm["x"] / 2
    cy = room_cm["y"] / 2
    # body offsets (x_lat, y_fwd, z_up) in cm relative to the sit point
    pose = {
        0:  ( 0,  4, 50),    # nose       — slight forward tilt of the head
        1:  (-2,  3, 52),    # left_eye
        2:  ( 2,  3, 52),    # right_eye
        3:  (-4,  0, 49),    # left_ear
        4:  ( 4,  0, 49),    # right_ear
        5:  (-9,  0, 38),    # left_shoulder
        6:  ( 9,  0, 38),    # right_shoulder
        7:  (-12,  4, 28),   # left_elbow
        8:  ( 12,  4, 28),   # right_elbow
        9:  (-12, 10, 16),   # left_wrist
       10:  ( 12, 10, 16),   # right_wrist
       11:  (-7,  0, 12),    # left_hip
       12:  ( 7,  0, 12),    # right_hip
       13:  (-9, 14, 14),    # left_knee  (legs folded forward)
       14:  ( 9, 14, 14),    # right_knee
       15:  (-9, 24,  4),    # left_ankle (feet near the floor)
       16:  ( 9, 24,  4),    # right_ankle
    }
    return {k: (cx + dx, cy + dy, dz) for k, (dx, dy, dz) in pose.items()}


def median_kp_errors():
    """Per-keypoint median 3D error in cm from the validation set
    (body keypoints only — KEPT_KP_IDX)."""
    kp = pd.read_hdf(CURATED, "keypoints")
    body = kp[kp["kp_idx"].isin(KEPT_KP_IDX)]
    de = (body.dropna(subset=["x_3d"])
              .drop_duplicates(["session","model","frame","animal","kp_idx"]))
    gt = de[de["model"]=="GT"].drop(columns="model")
    rt = de[de["model"]=="RT"].drop(columns="model")
    m = gt.merge(rt, on=["session","frame","animal","kp_idx"],
                 suffixes=("_gt","_rt"))
    dx = (m["x_3d_rt"] - m["x_3d_gt"]) * MM_TO_CM
    dy = (m["y_3d_rt"] - m["y_3d_gt"]) * MM_TO_CM
    dz = (m["z_3d_rt"] - m["z_3d_gt"]) * MM_TO_CM
    m["err_cm"] = np.sqrt(dx**2 + dy**2 + dz**2)
    return m.groupby("kp_idx")["err_cm"].median().to_dict()


def draw_canonical_pose_with_error(ax):
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]
    pose   = canonical_sitting_pose(ROOM_CM)
    errors = median_kp_errors()

    # Floor patch + room wireframe
    xs = np.array([[0, rx], [0, rx]])
    ys = np.array([[0, 0],  [ry, ry]])
    zs = np.zeros((2, 2))
    ax.plot_surface(xs, ys, zs, color="#88aa88", alpha=0.07,
                    linewidth=0, antialiased=False, shade=False)
    corners = [(0,0,0),(rx,0,0),(rx,ry,0),(0,ry,0),
               (0,0,rz),(rx,0,rz),(rx,ry,rz),(0,ry,rz)]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]),
                color="#666", lw=0.5, alpha=0.6)

    bone_color = PAL_WONG["Elm"]

    # Bones (full 17-keypoint COCO skeleton)
    for a, b in _FULL_COCO_PAIRS:
        if a in pose and b in pose:
            xs = [pose[a][0], pose[b][0]]
            ys = [pose[a][1], pose[b][1]]
            zs = [pose[a][2], pose[b][2]]
            ax.plot(xs, ys, zs, color=bone_color, lw=2.2, alpha=0.95)

    # Joints
    for k, (x, y, z) in pose.items():
        ax.scatter([x], [y], [z], s=22, color=bone_color,
                   edgecolors="black", linewidths=0.5, depthshade=False)

    # Uncertainty spheres ONLY on body keypoints we have errors for
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    sx = np.cos(u) * np.sin(v)
    sy = np.sin(u) * np.sin(v)
    sz = np.cos(v)
    for k, r in errors.items():
        if k not in pose or r <= 0:
            continue
        x, y, z = pose[k]
        ax.plot_surface(x + sx*r, y + sy*r, z + sz*r,
                        color=bone_color, alpha=0.22,
                        linewidth=0, antialiased=False, shade=False)

    ax.set_xlim(0, rx); ax.set_ylim(0, ry); ax.set_zlim(0, rz)
    ax.set_box_aspect((rx, ry, rz))
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_title("Per-keypoint error envelope around a canonical sitting pose")


def draw_distance(ax, dist_df):
    if dist_df.empty: return
    df = remove_outliers(dist_df, "dist_cm", ["session"])
    sns.violinplot(data=df, x="session", y="dist_cm",
                   inner=None, cut=0, color="#cce0f0",
                   linewidth=0.6, ax=ax)
    sns.stripplot(data=df, x="session", y="dist_cm",
                  color="#234", size=1.4, alpha=0.45, jitter=0.20, ax=ax)
    ax.set_ylabel("Elm ↔ Jok distance (cm)")
    ax.set_xlabel("session")
    ax.set_title("Inter-animal distance per session")
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45); lbl.set_ha("right")
    sns.despine(ax=ax)


# ─── Compose ─────────────────────────────────────────────────────────────────
def _save(fig, stem, subdir=None):
    out_dir = OUT_DIR / subdir if subdir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{stem}.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"  saved → {out}")
    plt.close(fig)


def main():
    print("Loading data…")
    df_2d, n2d_frames = load_validation_2d()
    df_3d, n3d_frames = load_validation_3d()
    centroids = load_centroids()
    dist_df   = inter_animal_distance(centroids)
    print(f"  2D error rows:     {len(df_2d)}  ({n2d_frames} GT frames)")
    print(f"  3D error rows:     {len(df_3d)}  ({n3d_frames} GT frames)")
    print(f"  centroid frames:   {len(centroids)}")
    print(f"  inter-animal dist: {len(dist_df)} (across {dist_df['session'].nunique()} sessions)")

    print("\nWriting standalone panels…")

    # ── ERRORS — 2D per camera + 3D per animal ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(W_SINGLE * 2.0, W_SINGLE * 0.85),
                             gridspec_kw={"wspace": 0.35,
                                          "width_ratios": [1.6, 0.8]})
    draw_2d_error_per_cam(axes[0], df_2d, n2d_frames)
    draw_3d_error_per_animal(axes[1], df_3d, n3d_frames)
    _save(fig, "talk_errors", subdir="validation")

    # ── SPATIAL — 3D voxels ──────────────────────────────────────────────────
    fig = plt.figure(figsize=(W_SINGLE * 1.1, W_SINGLE * 1.1))
    ax  = fig.add_subplot(projection="3d")
    draw_3d_voxels(ax, centroids)
    _save(fig, "talk_spatial", subdir="spatial")

    # ── SOCIAL — inter-animal distance ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(W_SINGLE * 1.4, W_SINGLE * 0.85))
    draw_distance(ax, dist_df)
    _save(fig, "talk_social", subdir="behavioral")

    # ── SKELETON — canonical sitting pose with per-keypoint error envelope ──
    fig = plt.figure(figsize=(W_SINGLE * 1.3, W_SINGLE * 1.2))
    ax  = fig.add_subplot(projection="3d")
    draw_canonical_pose_with_error(ax)
    _save(fig, "talk_skeleton_error", subdir="validation")


if __name__ == "__main__":
    main()
