#!/usr/bin/env python3
"""
Datalus Calibration — Point Picker (Step 3.5)

Interactive tool: click on a known room point in ≥2 camera images.
The script triangulates its 3D position in calibration space and appends
it to world_registration.csv.  You then fill in the real_x/y/z columns
from your physical measurements before running Step 4.

Usage:
  python3 DatulusCalib_PointPicker.py

Controls:
  Left-click   — mark this camera's 2D observation of the current point
  U            — undo last click for current point
  Enter/N      — triangulate current point and start next point
  Q/Escape     — quit and save CSV
"""

import sys
import os
import csv
import numpy as np
import cv2
import matplotlib
matplotlib.use("MacOSX")         # native macOS backend, no Tk dependency
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─── CONFIG ───────────────────────────────────────────────────────────────────

COLMAP_POSES_NPZ = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/colmap_poses.npz"
INTRINSICS_NPZ   = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/intrinsics.npz"
FRAMES_ROOT      = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/frames"
OUTPUT_CSV       = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/world_registration.csv"

CAMERA_IDS       = ["102", "108", "113", "117"]
REFERENCE_CAM    = "108"    # must be included in every triangulation for scale consistency

# Which frame to display for each camera (pick one with good room visibility)
# Set to None to auto-pick the middle frame
DISPLAY_FRAME    = None

MAX_REPROJ_ERROR = 20.0     # px — points above this threshold are rejected

# ─── HELPERS ──────────────────────────────────────────────────────────────────

def load_data():
    poses_raw = np.load(COLMAP_POSES_NPZ)
    intr_raw  = np.load(INTRINSICS_NPZ)

    poses      = {}
    intrinsics = {}
    for cam_id in CAMERA_IDS:
        if f"{cam_id}_R" in poses_raw:
            poses[cam_id] = (poses_raw[f"{cam_id}_R"], poses_raw[f"{cam_id}_T"].reshape(3,1))
        if f"{cam_id}_K" in intr_raw:
            intrinsics[cam_id] = {
                "K":    intr_raw[f"{cam_id}_K"],
                "dist": intr_raw[f"{cam_id}_dist"],
            }
    return poses, intrinsics


def load_display_image(cam_id):
    cam_dir = Path(FRAMES_ROOT) / cam_id
    frames  = sorted(cam_dir.glob("*.png"))
    if not frames:
        print(f"  [WARN] No frames found for cam {cam_id}")
        return None
    if DISPLAY_FRAME is not None:
        idx = min(DISPLAY_FRAME, len(frames)-1)
    else:
        idx = len(frames) // 2
    img = cv2.imread(str(frames[idx]))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def projection_matrix(K, R, T):
    """Build 3×4 projection matrix P = K @ [R | T]."""
    return K @ np.hstack([R, T.reshape(3,1)])


def triangulate_dlt(P_list, pts2d):
    """
    DLT triangulation from ≥2 (P, (x,y)) pairs.
    Returns 3D point in calibration space.
    """
    rows = []
    for P, (x, y) in zip(P_list, pts2d):
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def undistort_point(pt, K, dist):
    """Undistort a single 2D pixel point."""
    pts = np.array([[pt]], dtype=np.float64)
    und = cv2.undistortPoints(pts, K, dist, P=K)
    return und[0, 0]


def reprojection_error(X3d, P, pt2d):
    x = P @ np.append(X3d, 1.0)
    x = x[:2] / x[2]
    return np.linalg.norm(x - np.array(pt2d))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DATALUS — POINT PICKER (world registration helper)")
    print("=" * 60)

    poses, intrinsics = load_data()
    active_cams = [c for c in CAMERA_IDS if c in poses and c in intrinsics]
    print(f"  Cameras with poses: {active_cams}")

    # Build projection matrices
    P_map = {}
    for cam_id in active_cams:
        R, T = poses[cam_id]
        K    = intrinsics[cam_id]["K"]
        P_map[cam_id] = projection_matrix(K, R, T)

    # Load display images
    images = {}
    for cam_id in active_cams:
        img = load_display_image(cam_id)
        if img is not None:
            images[cam_id] = img
        else:
            print(f"  [WARN] Could not load image for cam {cam_id}")
    if not images:
        print("[ERROR] No images loaded.")
        sys.exit(1)

    # ── Set up matplotlib figure ───────────────────────────────────────────────
    n_cams = len(images)
    ncols  = 2
    nrows  = (n_cams + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 7 * nrows))
    axes = np.array(axes).flatten()

    # Fixed layout — never recalculates, so images don't shift on click
    fig.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.07,
                        hspace=0.08, wspace=0.04)

    cam_axes = {}
    for i, cam_id in enumerate(images.keys()):
        ax = axes[i]
        ax.imshow(images[cam_id])
        ax.set_title(f"cam {cam_id}", fontsize=12)
        ax.axis("off")
        cam_axes[cam_id] = ax

    # Hide unused subplots
    for j in range(n_cams, len(axes)):
        axes[j].set_visible(False)

    # State
    all_points   = []
    current      = {"clicks": {}, "markers": {}}
    point_counter = [1]

    status_ax = fig.add_axes([0.0, 0.0, 1.0, 0.06])
    status_ax.set_facecolor("#f0f0f0")
    status_ax.axis("off")
    status_text = status_ax.text(0.5, 0.5, "", ha="center", va="center",
                                  fontsize=11, transform=status_ax.transAxes)

    def update_status():
        n_clicks = len(current["clicks"])
        status_text.set_text(
            f"Point {point_counter[0]} — clicked in {n_clicks} camera(s) "
            f"({', '.join(current['clicks'].keys())})  |  "
            f"[Enter] triangulate & next   [U] undo   [Q] quit & save"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is None or event.button != 1:
            return
        # Find which camera was clicked
        clicked_cam = None
        for cam_id, ax in cam_axes.items():
            if event.inaxes == ax:
                clicked_cam = cam_id
                break
        if clicked_cam is None:
            return

        x, y = event.xdata, event.ydata

        # Remove previous marker for this cam if re-clicking
        if clicked_cam in current["markers"]:
            try:
                current["markers"][clicked_cam].remove()
            except Exception:
                pass

        marker, = cam_axes[clicked_cam].plot(
            x, y, "r+", markersize=18, markeredgewidth=2.5)
        current["clicks"][clicked_cam]  = (x, y)
        current["markers"][clicked_cam] = marker

        # Label
        cam_axes[clicked_cam].set_title(
            f"cam {clicked_cam}  ✓ ({x:.0f}, {y:.0f})", fontsize=11, color="green")

        update_status()

    def on_key(event):
        if event.key in ("enter", "n"):
            accept_point()
        elif event.key == "u":
            undo_last()
        elif event.key in ("q", "escape"):
            save_and_quit()

    def accept_point():
        if len(current["clicks"]) < 2:
            status_text.set_text("Need clicks in ≥2 cameras before triangulating.")
            fig.canvas.draw_idle()
            return

        if REFERENCE_CAM not in current["clicks"]:
            status_text.set_text(
                f"⚠  Must include cam {REFERENCE_CAM} (reference) in every point "
                f"for scale consistency.  Add a click in cam {REFERENCE_CAM} first.")
            fig.canvas.draw_idle()
            return

        # Undistort clicks, collect projection matrices
        P_list = []
        pts_ud = []
        for cam_id, (x, y) in current["clicks"].items():
            K    = intrinsics[cam_id]["K"]
            dist = intrinsics[cam_id]["dist"]
            xu, yu = undistort_point((x, y), K, dist)
            P_list.append(P_map[cam_id])
            pts_ud.append((xu, yu))

        X3d = triangulate_dlt(P_list, pts_ud)

        # Reprojection errors
        errs = []
        for cam_id, (x, y) in current["clicks"].items():
            err = reprojection_error(X3d, P_map[cam_id], (x, y))
            errs.append(err)
        mean_err = np.mean(errs)

        name = f"point_{point_counter[0]:02d}"
        print(f"  {name}: X=[{X3d[0]:.4f}, {X3d[1]:.4f}, {X3d[2]:.4f}]  "
              f"reproj_err={mean_err:.1f} px  (cams: {list(current['clicks'].keys())})")

        if mean_err > MAX_REPROJ_ERROR:
            print(f"  ✗ REJECTED (reproj_err {mean_err:.1f} > {MAX_REPROJ_ERROR} px) "
                  f"— clicks may not be on the same physical point. Try again.")
            status_text.set_text(
                f"✗ Rejected: reproj_err={mean_err:.1f} px (max {MAX_REPROJ_ERROR}).  "
                f"Clicks don't match — clear and try again.  [U] to undo last click.")
            # Draw rejected markers in red
            for cam_id, (x, y) in current["clicks"].items():
                cam_axes[cam_id].plot(x, y, "rx", markersize=14, markeredgewidth=2.5)
                cam_axes[cam_id].set_title(f"cam {cam_id}", fontsize=12, color="black")
            current["clicks"].clear()
            current["markers"].clear()
            fig.canvas.draw_idle()
            return

        all_points.append({
            "name":       name,
            "colmap_xyz": X3d,
            "n_cams":     len(current["clicks"]),
            "reproj_err": mean_err,
        })

        # Draw accepted point markers in blue
        for cam_id, (x, y) in current["clicks"].items():
            cam_axes[cam_id].plot(x, y, "bs", markersize=10, alpha=0.7)
            cam_axes[cam_id].annotate(
                name, (x, y), textcoords="offset points", xytext=(6, 6),
                fontsize=8, color="blue")
            cam_axes[cam_id].set_title(f"cam {cam_id}", fontsize=12, color="black")

        # Reset for next point
        point_counter[0] += 1
        current["clicks"].clear()
        current["markers"].clear()
        update_status()

    def undo_last():
        if not current["clicks"]:
            return
        last_cam = list(current["clicks"].keys())[-1]
        try:
            current["markers"][last_cam].remove()
        except Exception:
            pass
        del current["clicks"][last_cam]
        del current["markers"][last_cam]
        cam_axes[last_cam].set_title(f"cam {last_cam}", fontsize=12, color="black")
        update_status()

    def save_and_quit():
        if not all_points:
            print("  No points recorded — CSV not written.")
            plt.close("all")
            return

        write_header = not os.path.exists(OUTPUT_CSV)
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "colmap_x", "colmap_y", "colmap_z",
                             "real_x_mm", "real_y_mm", "real_z_mm"])
            for pt in all_points:
                X = pt["colmap_xyz"]
                writer.writerow([pt["name"],
                                  f"{X[0]:.6f}", f"{X[1]:.6f}", f"{X[2]:.6f}",
                                  "", "", ""])

        print(f"\n  Saved {len(all_points)} points → {OUTPUT_CSV}")
        print("  Fill in real_x_mm / real_y_mm / real_z_mm from your measurements,")
        print("  then run DatulusCalib_Step4_WorldReg.py.")
        plt.close("all")

    # Connect events
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)

    update_status()
    fig.suptitle(
        f"Click the SAME physical point in cam {REFERENCE_CAM} + ≥1 other camera, "
        f"then press Enter.\n"
        "Pick 4+ well-distributed points. reproj_err > "
        f"{MAX_REPROJ_ERROR:.0f} px = rejected (re-click more carefully).",
        fontsize=11, y=1.01
    )

    print("\n  Instructions:")
    print(f"  1. ALWAYS click in cam {REFERENCE_CAM} (reference) + ≥1 other camera per point")
    print("  2. Click on a sharp, identifiable feature (corner, joint, bracket end)")
    print("  3. Press Enter to triangulate — rejected if reproj_err > "
          f"{MAX_REPROJ_ERROR:.0f} px")
    print("  4. Repeat for ≥4 points spread around the room")
    print("  5. Press Q to save and quit")
    print("\n  Good reference points: cage corners, shelf ends, log tips, pipe joints\n")

    plt.show()


if __name__ == "__main__":
    main()
