#!/usr/bin/env python3
"""
HomeCagePaper · make_poster_cage.py

Interactive rotatable 3D cage view matching the rotating video style.
Shows: room wireframe, occupancy scatter (shuffled single-call),
camera positions as red triangles, and a black scale-reference monkey.

Run:  python3 HomeCagePaper/make_poster_cage.py
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import PAL_WONG, ANIMALS, MM_TO_CM, ROOM_CM

CURATED = HERE / "curated" / "master.h5"

mpl.rcParams.update({
    "font.family":     "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":       9,
})

CAM_POS = {
    "102": (30, 326, 54), "108": (185, 0, 248),
    "113": (5, 0, 55),    "117": (208, 307, 255),
}

BONES_17 = [(0,1),(0,2),(1,3),(2,4),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)]


def load_centroids():
    df = pd.read_hdf(CURATED, "centroids")
    return df.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})


def main():
    print("Loading centroids…")
    ct = load_centroids()
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]

    fig = plt.figure(figsize=(8, 8), dpi=120, facecolor="white")
    ax  = fig.add_subplot(projection="3d", facecolor="white")

    # ── Room wireframe (same as video) ───────────────────────────────────────
    corners = [(0,0,0),(rx,0,0),(rx,ry,0),(0,ry,0),
               (0,0,rz),(rx,0,rz),(rx,ry,rz),(0,ry,rz)]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]),
                color="#666", lw=0.6, alpha=0.8)

    # ── Occupancy scatter (single call, shuffled, same as video) ─────────────
    all_x, all_y, all_z, all_c = [], [], [], []
    for animal, sub in ct.groupby("animal"):
        xs = sub["x"].clip(0, rx).values
        ys = sub["y"].clip(0, ry).values
        zs = sub["z"].clip(0, rz).values
        c  = mpl.colors.to_rgba(PAL_WONG.get(animal, "#888"), alpha=0.15)
        all_x.append(xs); all_y.append(ys); all_z.append(zs)
        all_c.extend([c] * len(xs))
    all_x = np.concatenate(all_x)
    all_y = np.concatenate(all_y)
    all_z = np.concatenate(all_z)
    idx = np.random.default_rng(0).permutation(len(all_x))
    ax.scatter(all_x[idx], all_y[idx], all_z[idx], s=3,
               c=[all_c[i] for i in idx], edgecolors="none")

    # ── Camera positions (red triangles, same as video) ──────────────────────
    for cid, (cx, cy, cz) in CAM_POS.items():
        ax.scatter([cx], [cy], [cz], s=60, color="#ff4444",
                   marker="^", edgecolors="black", linewidths=0.5,
                   depthshade=False, zorder=10)

    # ── Scale-reference monkey (black, floor centre) ─────────────────────────
    mcx, mcy = rx/2, ry/2
    offsets = {
        0:  (0,  4, 50),  1: (-2,  3, 52),  2: ( 2,  3, 52),
        3: (-4,  0, 49),  4: ( 4,  0, 49),
        5: (-9,  0, 38),  6: ( 9,  0, 38),
        7:(-12,  4, 28),  8: (12,  4, 28),
        9:(-12, 10, 16), 10: (12, 10, 16),
       11: (-7,  0, 12), 12: ( 7,  0, 12),
       13: (-9, 14, 14), 14: ( 9, 14, 14),
       15: (-9, 24,  4), 16: ( 9, 24,  4),
    }
    skel = {k: (mcx+dx, mcy+dy, dz) for k, (dx, dy, dz) in offsets.items()}
    for a, b in BONES_17:
        ax.plot([skel[a][0], skel[b][0]],
                [skel[a][1], skel[b][1]],
                [skel[a][2], skel[b][2]],
                color="#222", lw=1.2, alpha=0.85)
    for k, (x, y, z) in skel.items():
        ax.scatter([x], [y], [z], s=12, color="#222",
                   edgecolors="white", linewidths=0.3,
                   depthshade=False, zorder=5)

    # ── Axes (same range as video) ───────────────────────────────────────────
    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_xlim(0, rx); ax.set_ylim(0, ry); ax.set_zlim(0, rz)
    ax.set_box_aspect((rx, ry, rz))

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [plt.Line2D([0],[0], marker="o", ls="", markersize=6,
                          color=PAL_WONG[a], label=a) for a in ANIMALS]
    handles.append(plt.Line2D([0],[0], marker="^", ls="", markersize=6,
                              color="#ff4444", label="Camera"))
    handles.append(plt.Line2D([0],[0], marker="o", ls="", markersize=5,
                              color="#222", label="Scale reference"))
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    # ── Starting view (same as video) ────────────────────────────────────────
    ax.view_init(elev=25, azim=-55)

    out = HERE / "figures" / "poster_cage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
