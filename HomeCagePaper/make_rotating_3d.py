#!/usr/bin/env python3
"""
Render a rotating 360° video of the 3D occupancy scatter.
Output: figures/spatial/rotating_3d.mp4
"""
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import apply_style, PAL_WONG, ANIMALS, KEPT_KP_IDX, MM_TO_CM, ROOM_CM
apply_style()

CURATED = HERE / "curated" / "master.h5"
OUT     = HERE / "_figures" / "spatial" / "rotating_3d.mp4"

FPS        = 30
DURATION_S = 12       # one full rotation
N_FRAMES   = FPS * DURATION_S


def load_centroids():
    df = pd.read_hdf(CURATED, "centroids")
    df = df.rename(columns={"x_cm": "x", "y_cm": "y", "z_cm": "z"})
    return df[["session", "animal", "frame", "x", "y", "z"]]


def main():
    print("Loading centroids…")
    ct = load_centroids()
    rx, ry, rz = ROOM_CM["x"], ROOM_CM["y"], ROOM_CM["z"]

    fig = plt.figure(figsize=(8, 8), dpi=120)
    ax  = fig.add_subplot(projection="3d")

    # Room wireframe
    corners = [(0,0,0),(rx,0,0),(rx,ry,0),(0,ry,0),
               (0,0,rz),(rx,0,rz),(rx,ry,rz),(0,ry,rz)]
    edges = [(0,1),(1,2),(2,3),(3,0),
             (4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    for a, b in edges:
        ax.plot(*zip(corners[a], corners[b]),
                color="#666", lw=0.6, alpha=0.8)

    # Combine all points into ONE scatter call so matplotlib depth-sorts
    # individual points rather than flipping entire layers.
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
    # Shuffle so neither animal consistently renders on top
    idx = np.random.default_rng(0).permutation(len(all_x))
    ax.scatter(all_x[idx], all_y[idx], all_z[idx], s=3,
               c=[all_c[i] for i in idx], edgecolors="none")

    # Camera positions
    CAM_POS = {
        "102": (30, 326, 54), "108": (185, 0, 248),
        "113": (5, 0, 55),    "117": (208, 307, 255),
    }
    for cid, (cx, cy, cz) in CAM_POS.items():
        ax.scatter([cx], [cy], [cz], s=60, color="#ff4444",
                   marker="^", edgecolors="black", linewidths=0.5,
                   depthshade=False, zorder=10)
        ax.text(cx, cy, cz + 12, cid, fontsize=7, ha="center",
                color="#ff4444", fontweight="bold")

    ax.set_xlabel("X (cm)"); ax.set_ylabel("Y (cm)"); ax.set_zlabel("Z (cm)")
    ax.set_xlim(0, rx); ax.set_ylim(0, ry); ax.set_zlim(0, rz)
    ax.set_box_aspect((rx, ry, rz))
    handles = [plt.Line2D([0],[0], marker="o", ls="", markersize=6,
                          color=PAL_WONG[a], label=a) for a in ANIMALS]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    # Render rotating frames as PNGs to a temp dir, then stitch with ffmpeg.
    # (cv2 VideoWriter with mp4v was producing un-finalized containers on macOS.)
    import shutil, subprocess, tempfile
    tmp_dir = Path(tempfile.mkdtemp(prefix="rot3d_"))
    print(f"Rendering {N_FRAMES} frames into {tmp_dir} …")
    try:
        for i in range(N_FRAMES):
            azim = 360 * i / N_FRAMES
            ax.view_init(elev=25, azim=azim)
            fig.savefig(tmp_dir / f"f{i:04d}.png", dpi=120,
                        facecolor=fig.get_facecolor())
            if i % 60 == 0:
                print(f"  frame {i}/{N_FRAMES}")
        plt.close(fig)

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install via `brew install ffmpeg`.")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg, "-y", "-framerate", str(FPS),
            "-i", str(tmp_dir / "f%04d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-crf", "20", "-preset", "medium",
            str(OUT),
        ]
        print("ffmpeg encode …")
        subprocess.run(cmd, check=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\nSaved → {OUT}  ({OUT.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
