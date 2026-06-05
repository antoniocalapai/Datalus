#!/usr/bin/env python3
"""
HomeCagePaper · Section 2 — Spatial analysis

Loads curated/master.csv, generates publication-ready PNGs in
figures/spatial/. Uses RT-predicted 3D positions only.

Functional, top-to-bottom in publication order. Re-runs overwrite.

Run:
    python3 HomeCagePaper/curate_data.py     # once
    python3 HomeCagePaper/spatial_analysis.py
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
from _style import (                                        # noqa: E402
    apply_style,
    W_SINGLE, W_DOUBLE, PAL_WONG,
    ANIMALS, KEPT_KP_IDX, ROOM_CM, MM_TO_CM,
)

CURATED = HERE / "curated" / "master.h5"
OUT_DIR = HERE / "_figures" / "spatial"
TBL_DIR = HERE / "tables"  / "spatial"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)

apply_style()

VOXEL_CM = 25.0   # ~half a macaque body length


# ─── 1. Load and reduce to per-frame centroids ───────────────────────────────
def load_centroids():
    """Per (session, animal, frame) trunk centroid (cm) — pre-computed by
    curate_data.py and stored in master.h5 under /centroids."""
    df = pd.read_hdf(CURATED, "centroids")
    df = df.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})
    return df[["session", "animal", "frame", "x", "y", "z"]].sort_values(
        ["session", "animal", "frame"]).reset_index(drop=True)


# ─── 2. Save helper ──────────────────────────────────────────────────────────
def _save(fig, stem):
    out = OUT_DIR / f"{stem}.png"
    fig.savefig(out, dpi=600)
    print(f"  {stem}.png")
    plt.close(fig)


# ─── 3. Panel drawers ────────────────────────────────────────────────────────
def _draw_room(ax):
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]
    corners = [(0,0,0),(rx,0,0),(rx,ry,0),(0,ry,0),
               (0,0,rz),(rx,0,rz),(rx,ry,rz),(0,ry,rz)]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]),
                color="#666", lw=0.6, alpha=0.8)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_xlim(0, rx); ax.set_ylim(0, ry); ax.set_zlim(0, rz)
    ax.set_box_aspect((rx, ry, rz))


def draw_3d_scatter(ax, centroids):
    _draw_room(ax)
    for animal, sub in centroids.groupby("animal"):
        xs = sub["x"].clip(0, ROOM_CM["x"])
        ys = sub["y"].clip(0, ROOM_CM["y"])
        zs = sub["z"].clip(0, ROOM_CM["z"])
        ax.scatter(xs, ys, zs, s=4, alpha=0.20,
                   color=PAL_WONG.get(animal, "#888"),
                   label=animal, edgecolors="none")
    ax.set_title("3D occupancy (scatter)")
    ax.legend(loc="upper left", fontsize=6)


def draw_3d_voxels(ax, centroids):
    _draw_room(ax)
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]
    nx = int(np.ceil(rx / VOXEL_CM))
    ny = int(np.ceil(ry / VOXEL_CM))
    nz = int(np.ceil(rz / VOXEL_CM))
    x_e = np.arange(nx + 1) * VOXEL_CM
    y_e = np.arange(ny + 1) * VOXEL_CM
    z_e = np.arange(nz + 1) * VOXEL_CM
    X, Y, Z = np.meshgrid(x_e, y_e, z_e, indexing="ij")
    for animal, sub in centroids.groupby("animal"):
        grid = np.zeros((nx, ny, nz), dtype=int)
        if not sub.empty:
            ix = np.clip((sub["x"] / VOXEL_CM).astype(int), 0, nx - 1)
            iy = np.clip((sub["y"] / VOXEL_CM).astype(int), 0, ny - 1)
            iz = np.clip((sub["z"] / VOXEL_CM).astype(int), 0, nz - 1)
            for a_, b_, c_ in zip(ix, iy, iz):
                grid[a_, b_, c_] += 1
        if not grid.any():
            continue
        filled = grid > 0
        max_c  = grid.max()
        alpha  = 0.20 + 0.65 * (grid / max_c)
        rgba = np.zeros(filled.shape + (4,))
        base = mpl.colors.to_rgb(PAL_WONG.get(animal, "#888"))
        rgba[..., 0] = base[0]; rgba[..., 1] = base[1]; rgba[..., 2] = base[2]
        rgba[..., 3] = np.where(filled, alpha, 0.0)
        ax.voxels(X, Y, Z, filled, facecolors=rgba,
                  edgecolor=(0, 0, 0, 0.20), linewidth=0.2, shade=True)
    ax.set_title(f"3D occupancy ({VOXEL_CM:.0f} cm voxels)")


def draw_height_heatmap(ax, centroids):
    n_bins = 30
    rz = ROOM_CM["z"]
    edges = np.linspace(0, rz, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    rows = []
    for animal in ANIMALS:
        sub = centroids[centroids["animal"] == animal]
        if sub.empty:
            rows.append(np.zeros(n_bins))
        else:
            h, _ = np.histogram(sub["z"], bins=edges)
            rows.append(h / max(1, h.sum()))
    mat = np.array(rows).T
    sns.heatmap(mat, ax=ax, cmap="rocket_r",
                yticklabels=[f"{c:.0f}" for c in centres],
                xticklabels=ANIMALS,
                cbar_kws={"label": "fraction of time", "shrink": 0.7})
    ax.invert_yaxis()
    for i, lbl in enumerate(ax.get_yticklabels()):
        lbl.set_visible(i % 5 == 0)
    ax.set_xlabel("")
    ax.set_ylabel("height (cm)")
    ax.set_title("Time at each height")


def draw_topdown(ax, centroids):
    rx, ry = ROOM_CM["x"], ROOM_CM["y"]
    ax.add_patch(plt.Rectangle((0, 0), rx, ry, ec="#666", fc="none", lw=0.6))
    for animal, sub in centroids.groupby("animal"):
        ax.scatter(sub["x"].clip(0, rx), sub["y"].clip(0, ry),
                   s=6, alpha=0.30,
                   color=PAL_WONG.get(animal, "#888"),
                   label=animal, edgecolors="none")
    ax.set_xlim(-5, rx + 5); ax.set_ylim(-5, ry + 5)
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Top-down occupancy")
    ax.legend(loc="upper right", fontsize=6)


# ─── 4. Statistics ───────────────────────────────────────────────────────────
def write_stats(centroids):
    rows = []
    for animal in ANIMALS:
        sub = centroids[centroids["animal"] == animal]
        if sub.empty:
            continue
        rows.append({
            "animal":     animal,
            "n_frames":   int(len(sub)),
            "x_med_cm":   float(sub["x"].median()),
            "y_med_cm":   float(sub["y"].median()),
            "z_med_cm":   float(sub["z"].median()),
            "z_p25_cm":   float(sub["z"].quantile(0.25)),
            "z_p75_cm":   float(sub["z"].quantile(0.75)),
            "z_min_cm":   float(sub["z"].min()),
            "z_max_cm":   float(sub["z"].max()),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TBL_DIR / "stats.csv", index=False)
    print("\nSpatial summary:")
    print(out.to_string(index=False))


# ─── 5. Figure builders (chronological) ──────────────────────────────────────
def build_figure1(centroids):
    """3D spatial occupancy — alpha-blended scatter and voxel decomposition."""
    print("\n── figure1 — 3D spatial occupancy ──")

    fig = plt.figure(figsize=(W_DOUBLE, W_DOUBLE * 0.55))
    ax_a = fig.add_subplot(1, 2, 1, projection="3d")
    ax_b = fig.add_subplot(1, 2, 2, projection="3d")
    draw_3d_scatter(ax_a, centroids)
    draw_3d_voxels(ax_b,  centroids)
    fig.subplots_adjust(wspace=0.10)
    _save(fig, "figure1")

    fig = plt.figure(figsize=(W_SINGLE, W_SINGLE))
    ax = fig.add_subplot(projection="3d")
    draw_3d_scatter(ax, centroids); _save(fig, "figure1_a")

    fig = plt.figure(figsize=(W_SINGLE, W_SINGLE))
    ax = fig.add_subplot(projection="3d")
    draw_3d_voxels(ax, centroids); _save(fig, "figure1_b")


def build_figure2(centroids):
    """Top-down (xy) occupancy."""
    print("\n── figure2 — Top-down occupancy ──")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE))
    draw_topdown(ax, centroids)
    _save(fig, "figure2")


def build_figure3(centroids):
    """Vertical use of the cage — height heatmap."""
    print("\n── figure3 — Height usage ──")
    fig, ax = plt.subplots(figsize=(W_SINGLE, W_SINGLE))
    draw_height_heatmap(ax, centroids)
    _save(fig, "figure3")


# ─── 6. main ─────────────────────────────────────────────────────────────────
def main():
    print(f"Loading {CURATED.name}")
    centroids = load_centroids()
    print(f"  centroids: {len(centroids)}  "
          f"({centroids.groupby('animal').size().to_dict()})")
    print(f"\nWriting figures to {OUT_DIR}")

    build_figure1(centroids)
    build_figure2(centroids)
    build_figure3(centroids)
    write_stats(centroids)

    print(f"\nDone. {len(list(OUT_DIR.glob('figure*.png')))} PNGs.")


if __name__ == "__main__":
    main()
