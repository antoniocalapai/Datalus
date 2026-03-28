#!/usr/bin/env python3
"""
Datalus — Interactive 3D Trajectory Visualizer (Plotly)

Loads all frames from Measurements/250711/2D_results/ for all 4 cameras,
triangulates the hip midpoint (avg of left_hip kp11 + right_hip kp12) per
frame per animal, and outputs a single self-contained HTML file with:

  - 3D trajectory lines colored by time via continuous colorscale
  - Moving current-position marker per animal
  - Frame scrubber slider (≤300 steps) + Play/Pause buttons
  - 4 camera positions as red diamond markers
  - Dark theme

Only the static trajectory lines are stored once; each slider step updates
only the two small position markers → compact HTML regardless of frame count.

Output: trajectory_output.html
"""

import cv2
import os
import re
import sys
import numpy as np
from pathlib import Path

import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR       = "/Users/acalapai/PycharmProjects/Datalus"
RESULTS_DIR    = f"{BASE_DIR}/Measurements/250711/2D_results"
YAML_DIR       = f"{BASE_DIR}/DatalusCalibration/yaml"
OUTPUT_HTML    = f"{BASE_DIR}/trajectory_output.html"

CAMERA_IDS     = ["102", "108", "113", "117"]
CONF_THRESH    = 0.3
HIP_INDICES    = [11, 12]   # left_hip, right_hip
N_SLIDER_STEPS = 300        # max slider positions

# Per-animal: (line colorscale, marker hex color)
PALETTES = {
    "Elm": ("Viridis", "#00eeff"),
    "Jok": ("Hot",     "#ff7700"),
}
FALLBACK_PALETTES = [("Blues", "#ccccff"), ("Greens", "#aaffaa")]


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration(yaml_dir, camera_ids):
    cams = {}
    for cam_id in camera_ids:
        p = Path(yaml_dir) / f"{cam_id}.yaml"
        if not p.exists():
            print(f"[ERROR] Missing YAML: {p}")
            sys.exit(1)
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        K    = fs.getNode("intrinsicMatrix").mat().T
        dist = fs.getNode("distortionCoefficients").mat().ravel()
        R    = fs.getNode("R").mat().T
        T    = fs.getNode("T").mat().reshape(3, 1)
        fs.release()
        cams[cam_id] = dict(K=K, dist=dist, R=R, T=T)
    return cams


# ── Parsing ───────────────────────────────────────────────────────────────────

def find_result_file(results_dir, cam_id):
    pat = re.compile(rf".*__{cam_id}_.*_2D_result\.txt")
    for p in Path(results_dir).glob("*.txt"):
        if pat.match(p.name):
            return p
    return None


def parse_results(path):
    """Returns {frame_int: {monkey_str: kps(17,3)}}"""
    data = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            cols = line.split()
            if len(cols) < 7 + 17 * 3:
                continue
            frame  = int(cols[0])
            monkey = cols[1]
            kps    = np.array(cols[7:7 + 17 * 3], dtype=np.float32).reshape(17, 3)
            data.setdefault(frame, {})[monkey] = kps
    return data


# ── Triangulation ─────────────────────────────────────────────────────────────

def _dlt(rays):
    """DLT from list of (cam_id, xu, yu, P[3x4])."""
    rows = []
    for _, xu, yu, P in rays:
        rows.append(xu * P[2] - P[0])
        rows.append(yu * P[2] - P[1])
    A = np.array(rows)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-10:
        return None
    return X[:3] / X[3]


def triangulate_point(obs_per_cam, cams):
    """
    obs_per_cam: {cam_id: (x_px, y_px)}
    Returns 3D point (3,) or None.
    """
    rays = []
    for cam_id, (x, y) in obs_per_cam.items():
        K = cams[cam_id]["K"]
        dist = cams[cam_id]["dist"]
        R = cams[cam_id]["R"]
        T = cams[cam_id]["T"]
        pt_ud = cv2.undistortPoints(
            np.array([[[x, y]]], dtype=np.float32), K, dist)[0, 0]
        P = np.hstack([R, T])
        rays.append((cam_id, pt_ud[0], pt_ud[1], P))

    if len(rays) < 2:
        return None

    X3 = _dlt(rays)
    if X3 is None:
        return None

    # Drop cameras where the point has non-positive depth (behind camera)
    good = [r for r in rays
            if (cams[r[0]]["R"] @ X3 + cams[r[0]]["T"].ravel())[2] > 1e-4]
    if len(good) < 2:
        return None
    if len(good) < len(rays):
        X3 = _dlt(good)
    return X3


def hip_midpoint_3d(kps_per_cam, cams):
    """
    Triangulates left_hip (11) and right_hip (12) independently,
    returns their 3D midpoint, or None if either fails.
    """
    pts3d = []
    for hip_idx in HIP_INDICES:
        obs = {}
        for cam_id, kps in kps_per_cam.items():
            x, y, c = kps[hip_idx]
            if c >= CONF_THRESH:
                obs[cam_id] = (float(x), float(y))
        if len(obs) < 2:
            return None
        p = triangulate_point(obs, cams)
        if p is None:
            return None
        pts3d.append(p)
    return np.mean(pts3d, axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading calibration...")
    cams = load_calibration(YAML_DIR, CAMERA_IDS)

    print("Loading 2D results...")
    all_data = {}
    for cam_id in CAMERA_IDS:
        p = find_result_file(RESULTS_DIR, cam_id)
        if p is None:
            print(f"  [WARN] No result file for cam {cam_id}")
            all_data[cam_id] = {}
        else:
            all_data[cam_id] = parse_results(p)
            n_det = sum(len(m) for m in all_data[cam_id].values())
            print(f"  cam {cam_id}: {len(all_data[cam_id])} frames, "
                  f"{n_det} detections  ← {p.name}")

    all_frames = sorted(set().union(*[d.keys() for d in all_data.values()]))
    print(f"\n  Total frames spanning any camera: {len(all_frames)}")

    # ── Triangulate hip midpoints ──────────────────────────────────────────────
    print("\nTriangulating hip midpoints...")
    trajectories = {}   # monkey -> list of [frame, x, y, z]

    for frame in all_frames:
        # Gather per-monkey detections across cameras for this frame
        monkey_cams = {}
        for cam_id in CAMERA_IDS:
            for monkey, kps in all_data[cam_id].get(frame, {}).items():
                monkey_cams.setdefault(monkey, {})[cam_id] = kps

        for monkey, kps_per_cam in monkey_cams.items():
            if len(kps_per_cam) < 2:
                continue
            p3d = hip_midpoint_3d(kps_per_cam, cams)
            if p3d is None:
                continue
            trajectories.setdefault(monkey, []).append(
                [frame, float(p3d[0]), float(p3d[1]), float(p3d[2])])

    if not trajectories:
        print("[ERROR] No trajectories computed. Check CONF_THRESH and YAML files.")
        sys.exit(1)

    for monkey, traj in sorted(trajectories.items()):
        print(f"  {monkey}: {len(traj):>5} positions  "
              f"(frames {traj[0][0]}–{traj[-1][0]})")

    # ── Assign palettes ───────────────────────────────────────────────────────
    monkey_names = sorted(trajectories.keys())
    palette_map  = {}
    fallback_idx = 0
    for monkey in monkey_names:
        if monkey in PALETTES:
            palette_map[monkey] = PALETTES[monkey]
        else:
            palette_map[monkey] = FALLBACK_PALETTES[fallback_idx % len(FALLBACK_PALETTES)]
            fallback_idx += 1

    # ── Camera centres ────────────────────────────────────────────────────────
    cam_centres = {
        cam_id: (-c["R"].T @ c["T"]).ravel().tolist()
        for cam_id, c in cams.items()
    }

    # ── Build static traces ───────────────────────────────────────────────────
    static_traces = []

    for monkey in monkey_names:
        arr = np.array(trajectories[monkey])   # (N, 4): frame, x, y, z
        cs, mc = palette_map[monkey]
        t_norm = ((arr[:, 0] - arr[0, 0]) /
                  max(float(arr[-1, 0] - arr[0, 0]), 1.0)).tolist()

        static_traces.append(go.Scatter3d(
            x=arr[:, 1].tolist(),
            y=arr[:, 2].tolist(),
            z=arr[:, 3].tolist(),
            mode="lines",
            name=monkey,
            line=dict(
                color=t_norm,
                colorscale=cs,
                width=4,
                cmin=0.0,
                cmax=1.0,
            ),
            hovertemplate=(
                f"<b>{monkey}</b><br>"
                "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}"
                "<extra></extra>"
            ),
        ))

    # Camera markers
    static_traces.append(go.Scatter3d(
        x=[C[0] for C in cam_centres.values()],
        y=[C[1] for C in cam_centres.values()],
        z=[C[2] for C in cam_centres.values()],
        mode="markers+text",
        name="Cameras",
        text=[f"cam{c}" for c in cam_centres.keys()],
        textposition="top center",
        textfont=dict(color="white", size=11),
        marker=dict(
            color="red",
            size=9,
            symbol="diamond",
            line=dict(color="white", width=1),
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}"
            "<extra></extra>"
        ),
    ))

    n_static = len(static_traces)

    # ── Animated traces: current-position markers ─────────────────────────────
    # One per animal; updated cheaply by each slider step.
    traj_arrays = {m: np.array(trajectories[m]) for m in monkey_names}

    anim_traces = []
    for monkey in monkey_names:
        arr = traj_arrays[monkey]
        _, mc = palette_map[monkey]
        # Initial: last known position (full trajectory shown)
        anim_traces.append(go.Scatter3d(
            x=[arr[-1, 1]],
            y=[arr[-1, 2]],
            z=[arr[-1, 3]],
            mode="markers+text",
            name=f"{monkey} (now)",
            text=[f"◉ {monkey}"],
            textposition="top center",
            textfont=dict(color=mc, size=12),
            marker=dict(
                color=mc,
                size=14,
                symbol="circle",
                line=dict(color="white", width=2),
            ),
            hovertemplate=(
                f"<b>{monkey} — current</b><br>"
                "X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}"
                "<extra></extra>"
            ),
        ))

    animated_indices = list(range(n_static, n_static + len(monkey_names)))
    all_traces = static_traces + anim_traces

    # ── Slider steps ──────────────────────────────────────────────────────────
    all_traj_frames = sorted({int(row[0]) for t in trajectories.values() for row in t})
    step_size   = max(1, len(all_traj_frames) // N_SLIDER_STEPS)
    slider_frames = all_traj_frames[::step_size]
    if all_traj_frames[-1] not in slider_frames:
        slider_frames.append(all_traj_frames[-1])

    print(f"\nBuilding {len(slider_frames)} slider steps...")

    plotly_frames = []
    for sf in slider_frames:
        frame_data = []
        for monkey in monkey_names:
            arr = traj_arrays[monkey]
            _, mc = palette_map[monkey]
            past = arr[arr[:, 0] <= sf]
            if len(past) == 0:
                frame_data.append(go.Scatter3d(x=[], y=[], z=[], mode="markers"))
            else:
                frame_data.append(go.Scatter3d(
                    x=[past[-1, 1]],
                    y=[past[-1, 2]],
                    z=[past[-1, 3]],
                    mode="markers+text",
                    text=[f"◉ {monkey}"],
                    textposition="top center",
                    textfont=dict(color=mc, size=12),
                    marker=dict(color=mc, size=14, symbol="circle",
                                line=dict(color="white", width=2)),
                ))
        plotly_frames.append(go.Frame(
            data=frame_data,
            traces=animated_indices,
            name=str(sf),
        ))

    # ── Slider UI ─────────────────────────────────────────────────────────────
    sliders = [{
        "active": len(slider_frames) - 1,
        "steps": [
            {
                "args": [
                    [str(sf)],
                    {"frame": {"duration": 0, "redraw": True},
                     "mode": "immediate",
                     "transition": {"duration": 0}},
                ],
                "label": str(sf),
                "method": "animate",
            }
            for sf in slider_frames
        ],
        "x": 0.05,
        "y": 0.02,
        "len": 0.88,
        "xanchor": "left",
        "yanchor": "bottom",
        "pad": {"b": 10, "t": 40},
        "bgcolor": "#2a2a4a",
        "bordercolor": "#555588",
        "borderwidth": 1,
        "tickcolor": "#aaaacc",
        "font": {"color": "white"},
        "currentvalue": {
            "prefix": "Frame: ",
            "visible": True,
            "xanchor": "center",
            "font": {"color": "white", "size": 13},
        },
        "transition": {"duration": 0},
    }]

    updatemenus = [{
        "type": "buttons",
        "showactive": False,
        "x": 0.0,
        "y": 0.0,
        "xanchor": "left",
        "yanchor": "bottom",
        "pad": {"b": 55, "r": 10},
        "bgcolor": "#1a1a3a",
        "bordercolor": "#555588",
        "font": {"color": "white"},
        "buttons": [
            {
                "label": "▶ Play",
                "method": "animate",
                "args": [None, {
                    "frame": {"duration": 60, "redraw": True},
                    "fromcurrent": True,
                    "mode": "immediate",
                    "transition": {"duration": 0},
                }],
            },
            {
                "label": "⏸ Pause",
                "method": "animate",
                "args": [[None], {
                    "frame": {"duration": 0, "redraw": False},
                    "mode": "immediate",
                }],
            },
        ],
    }]

    # ── Assemble figure ───────────────────────────────────────────────────────
    fig = go.Figure(
        data=all_traces,
        layout=go.Layout(
            title=dict(
                text="Monkey Hip Trajectories — session 250711<br>"
                     "<sup>Line color = time (early → late). "
                     "Drag slider or press Play to animate.</sup>",
                font=dict(color="white", size=16),
                x=0.5,
            ),
            scene=dict(
                xaxis=dict(
                    title="X",
                    backgroundcolor="#12122a",
                    gridcolor="#2a2a5a",
                    zerolinecolor="#3a3a6a",
                    color="white",
                ),
                yaxis=dict(
                    title="Y",
                    backgroundcolor="#12122a",
                    gridcolor="#2a2a5a",
                    zerolinecolor="#3a3a6a",
                    color="white",
                ),
                zaxis=dict(
                    title="Z",
                    backgroundcolor="#12122a",
                    gridcolor="#2a2a5a",
                    zerolinecolor="#3a3a6a",
                    color="white",
                ),
                bgcolor="#0d0d22",
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=0.9),
                    up=dict(x=0, y=0, z=1),
                ),
                aspectmode="data",
            ),
            paper_bgcolor="#0a0a1a",
            font=dict(color="white", family="monospace"),
            sliders=sliders,
            updatemenus=updatemenus,
            height=800,
            margin=dict(l=0, r=0, b=130, t=80),
            legend=dict(
                bgcolor="rgba(20,20,50,0.85)",
                bordercolor="#444477",
                borderwidth=1,
                font=dict(color="white", size=12),
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
            ),
        ),
        frames=plotly_frames,
    )

    # ── Write HTML ────────────────────────────────────────────────────────────
    print(f"Writing HTML → {OUTPUT_HTML}")
    fig.write_html(
        OUTPUT_HTML,
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True},
    )

    size_mb = Path(OUTPUT_HTML).stat().st_size / 1e6
    print(f"Done. File size: {size_mb:.2f} MB")
    print(f"\nOpen in browser:\n  open '{OUTPUT_HTML}'")
    os.system(f"open '{OUTPUT_HTML}'")


if __name__ == "__main__":
    main()
