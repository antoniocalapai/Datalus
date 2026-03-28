#!/usr/bin/env python3
"""
build_viewer.py — Generates datalus_viewer.html

Self-contained multi-view synchronized viewer:
  Left  : 2×2 grid of camera images with bounding boxes (Elm=blue, Jok=red)
  Right : 3D Plotly scene — room wireframe, cameras at physical positions,
          animal spheres
  Bottom: single frame slider + play/pause controlling both panels

Calibration: R from essmat pose correspondence, T = -R @ C_physical (mm).
             Loaded from DatalusCalibration/yaml/{cam_id}.yaml (ABT format).
             Camera centres = physical room positions from datalus_config.json.
"""

import base64, json, re
import cv2
import numpy as np
from itertools import combinations
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE         = Path("/Users/acalapai/PycharmProjects/Datalus")
CONFIG_JSON  = BASE / "datalus_config.json"
YAML_DIR     = BASE / "DatalusCalibration/yaml"
FRAMES_DIR   = BASE / "Measurements/250711/extracted_frames"
RESULTS_DIR  = BASE / "Measurements/250711/2D_results"
OUTPUT_HTML  = BASE / "datalus_viewer.html"

CAMERA_IDS    = ["102", "108", "113", "117"]
CONF_THRESH   = 0.3
IMG_SCALE     = 0.25
JPEG_QUALITY  = 40
ANIMAL_COLORS = {"Elm": "#3399ff", "Jok": "#ff5500"}


# ── Calibration ───────────────────────────────────────────────────────────────

def load_calibration(cfg):
    """
    Load K, dist, R, T from ABT YAMLs (stores K^T and R^T — transpose on read).
    T = -R @ C_physical  →  camera centre C = -R^T @ T = C_physical (mm, rel. cam102).
    Display position = C_relative + cam102_abs = absolute room mm.
    """
    cam102_abs = np.array([
        cfg["cameras"]["102"]["position_mm"]["x"],
        cfg["cameras"]["102"]["position_mm"]["y"],
        cfg["cameras"]["102"]["position_mm"]["z"],
    ], dtype=np.float64)

    cams = {}
    for cam_id in CAMERA_IDS:
        p  = YAML_DIR / f"{cam_id}.yaml"
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        K    = fs.getNode("intrinsicMatrix").mat().T.astype(np.float64)
        dist = fs.getNode("distortionCoefficients").mat().ravel().astype(np.float64)
        R    = fs.getNode("R").mat().T.astype(np.float64)
        T    = fs.getNode("T").mat().reshape(3, 1).astype(np.float64)
        fs.release()
        P     = K @ np.hstack([R, T])
        C_rel = (-R.T @ T).ravel()          # mm relative to cam102
        C_abs = C_rel + cam102_abs          # absolute room mm
        cams[cam_id] = dict(K=K, dist=dist, R=R, T=T, P=P,
                            C_rel=C_rel, C_abs=C_abs)
    return cams, cam102_abs


# ── Parsing ───────────────────────────────────────────────────────────────────

def find_result_file(cam_id):
    pat = re.compile(rf".*__{cam_id}_.*_2D_result\.txt")
    for p in RESULTS_DIR.glob("*.txt"):
        if pat.match(p.name):
            return p
    return None


def parse_results_for_frames(cam_id, wanted_frames: set):
    """Returns {frame: {monkey: {'bbox': [x1,y1,x2,y2], 'kps': (17,3)}}}"""
    p = find_result_file(cam_id)
    if p is None:
        return {}
    data = {}
    with open(p) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:
                continue
            cols = line.split()
            if len(cols) < 7 + 17 * 3:
                continue
            frame = int(cols[0])
            if frame not in wanted_frames:
                continue
            monkey = cols[1]
            bbox   = [float(cols[2]), float(cols[3]),
                      float(cols[4]), float(cols[5])]
            kps    = np.array(cols[7:7 + 51], dtype=np.float32).reshape(17, 3)
            data.setdefault(frame, {})[monkey] = {"bbox": bbox, "kps": kps}
    return data


# ── Triangulation ─────────────────────────────────────────────────────────────

def triangulate_hip(kps_per_cam: dict, cams: dict):
    """
    Triangulates left_hip (11) + right_hip (12) using pairwise
    cv2.triangulatePoints with P = K @ [R|T_essmat].
    Returns 3D midpoint in essmat world units (cam102 at origin), or None.
    """
    results = []
    for hip_idx in [11, 12]:
        obs = {}
        for cam_id, det in kps_per_cam.items():
            x, y, c = det["kps"][hip_idx]
            if c >= CONF_THRESH:
                obs[cam_id] = (float(x), float(y))
        if len(obs) < 2:
            continue

        pts3d_pairs = []
        for ca, cb in combinations(obs.keys(), 2):
            Pa = cams[ca]["P"]
            Pb = cams[cb]["P"]
            xa, ya = obs[ca]
            xb, yb = obs[cb]
            X4 = cv2.triangulatePoints(
                Pa, Pb,
                np.array([[xa], [ya]], dtype=np.float64),
                np.array([[xb], [yb]], dtype=np.float64),
            )
            if abs(X4[3]) < 1e-10:
                continue
            X3 = (X4[:3] / X4[3]).ravel()
            if not np.all(np.isfinite(X3)):
                continue
            # Cheirality: must be in front of both cameras
            Ra, Ta = cams[ca]["R"], cams[ca]["T"].ravel()
            Rb, Tb = cams[cb]["R"], cams[cb]["T"].ravel()
            if (Ra @ X3 + Ta)[2] > 0 and (Rb @ X3 + Tb)[2] > 0:
                pts3d_pairs.append(X3)

        if not pts3d_pairs:
            continue
        results.append(np.median(pts3d_pairs, axis=0))

    if len(results) < 2:
        return None
    return (results[0] + results[1]) / 2.0


# ── Image helpers ─────────────────────────────────────────────────────────────

def encode_image(img_path: Path) -> str:
    """Load, resize, JPEG-encode, return base64 data-URI."""
    img = cv2.imread(str(img_path))
    if img is None:
        return ""
    h, w = img.shape[:2]
    new_w = int(w * IMG_SCALE)
    new_h = int(h * IMG_SCALE)
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # BGR → JPEG bytes
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"



# ── Main ──────────────────────────────────────────────────────────────────────

def room_wireframe_trace(room):
    rx, ry, rz = room["x"], room["y"], room["z"]
    corners = [
        [0,0,0],[rx,0,0],[rx,ry,0],[0,ry,0],
        [0,0,rz],[rx,0,rz],[rx,ry,rz],[0,ry,rz],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [corners[a][0], corners[b][0], None]
        ys += [corners[a][1], corners[b][1], None]
        zs += [corners[a][2], corners[b][2], None]
    return {"type": "scatter3d", "x": xs, "y": ys, "z": zs,
            "mode": "lines", "name": "Room",
            "line": {"color": "#445566", "width": 2}, "hoverinfo": "skip"}


def main():
    cfg = json.loads(CONFIG_JSON.read_text())
    room = cfg["room_mm"]

    print("Loading calibration from YAMLs (R_essmat + T_physical)...")
    cams, cam102_abs = load_calibration(cfg)
    print("  Camera positions (absolute room mm):")
    for c in CAMERA_IDS:
        ca = cams[c]["C_abs"]
        print(f"    cam {c}: [{ca[0]:.0f}, {ca[1]:.0f}, {ca[2]:.0f}]")

    # ── Find shared frames ─────────────────────────────────────────────────────
    print("Scanning frame files...")
    cam_frame_sets = {}
    for cam_id in CAMERA_IDS:
        frames = set()
        for p in FRAMES_DIR.glob(f"{cam_id}_20250711_frame_*.png"):
            m = re.match(rf"{cam_id}_20250711_frame_(\d+)\.png", p.name)
            if m:
                frames.add(int(m.group(1)))
        cam_frame_sets[cam_id] = frames
        print(f"  cam {cam_id}: {len(frames)} image files")

    shared_frames = sorted(set.intersection(*cam_frame_sets.values()))
    print(f"  Shared frames: {len(shared_frames)}  ({shared_frames[0]}–{shared_frames[-1]})")

    # ── Load 2D results for shared frames ─────────────────────────────────────
    print("Loading 2D results...")
    wanted = set(shared_frames)
    results = {}   # cam_id -> {frame: {monkey: {bbox, kps}}}
    for cam_id in CAMERA_IDS:
        results[cam_id] = parse_results_for_frames(cam_id, wanted)
        n = sum(len(m) for m in results[cam_id].values())
        print(f"  cam {cam_id}: {len(results[cam_id])} frames with detections, "
              f"{n} total detections")

    # ── Triangulate per frame ─────────────────────────────────────────────────
    print("Triangulating hip midpoints...")
    positions3d = {}   # frame -> {monkey: [x_room, y_room, z_room]}

    for frame in shared_frames:
        monkey_cam_dets = {}
        for cam_id in CAMERA_IDS:
            for monkey, det in results[cam_id].get(frame, {}).items():
                monkey_cam_dets.setdefault(monkey, {})[cam_id] = det

        frame_pos = {}
        for monkey, kps_per_cam in monkey_cam_dets.items():
            if len(kps_per_cam) < 2:
                continue
            X3_rel = triangulate_hip(kps_per_cam, cams)
            if X3_rel is None:
                continue
            # Convert to absolute room mm and sanity-check within 2× room
            X3_abs = X3_rel + cam102_abs
            if (np.all(np.isfinite(X3_abs))
                    and -room["x"] < X3_abs[0] < 2*room["x"]
                    and -room["y"] < X3_abs[1] < 2*room["y"]
                    and -room["z"] < X3_abs[2] < 2*room["z"]):
                frame_pos[monkey] = X3_abs.tolist()

        positions3d[frame] = frame_pos

    n_with_pos = sum(1 for v in positions3d.values() if v)
    for animal in ["Elm", "Jok"]:
        cnt = sum(1 for v in positions3d.values() if animal in v)
        print(f"  {animal}: {cnt}/{len(shared_frames)} frames triangulated")

    # ── Encode images ─────────────────────────────────────────────────────────
    print(f"Encoding {len(shared_frames) * len(CAMERA_IDS)} images "
          f"(scale={IMG_SCALE}, q={JPEG_QUALITY})...")
    images = {cam_id: [] for cam_id in CAMERA_IDS}
    for frame in shared_frames:
        for cam_id in CAMERA_IDS:
            img_path = FRAMES_DIR / f"{cam_id}_20250711_frame_{frame:06d}.png"
            images[cam_id].append(encode_image(img_path))
    print("  Done encoding.")

    # ── Serialize detections (bbox only, for JS drawing) ─────────────────────
    # Structure: {frame: {monkey: {cam_id: [x1,y1,x2,y2]}}}
    det_js = {}
    for frame in shared_frames:
        fd = {}
        for cam_id in CAMERA_IDS:
            for monkey, det in results[cam_id].get(frame, {}).items():
                fd.setdefault(monkey, {})[cam_id] = det["bbox"]
        det_js[frame] = fd

    # ── Build Plotly initial figure data ──────────────────────────────────────
    cam_abs = {c: cams[c]["C_abs"].tolist() for c in CAMERA_IDS}

    def first_pos(animal):
        for frame in shared_frames:
            if animal in positions3d.get(frame, {}):
                return positions3d[frame][animal]
        return cam_abs["102"]

    plotly_traces = [
        room_wireframe_trace(room),
        {
            "type": "scatter3d",
            "x": [cam_abs[c][0] for c in CAMERA_IDS],
            "y": [cam_abs[c][1] for c in CAMERA_IDS],
            "z": [cam_abs[c][2] for c in CAMERA_IDS],
            "mode": "markers+text",
            "name": "Cameras",
            "text": [f"cam{c}" for c in CAMERA_IDS],
            "textposition": "top center",
            "textfont": {"color": "white", "size": 11},
            "marker": {"color": "red", "size": 8, "symbol": "diamond",
                       "line": {"color": "white", "width": 1}},
            "hovertemplate": "<b>%{text}</b><extra></extra>",
        },
    ]
    # Animal sphere traces — indices 2 and 3
    animal_trace_indices = {}
    for i, animal in enumerate(["Elm", "Jok"]):
        pos = first_pos(animal)
        color = ANIMAL_COLORS.get(animal, "#ffffff")
        plotly_traces.append({
            "type": "scatter3d",
            "x": [pos[0]], "y": [pos[1]], "z": [pos[2]],
            "mode": "markers+text",
            "name": animal,
            "text": [animal],
            "textposition": "top center",
            "textfont": {"color": color, "size": 13},
            "marker": {"color": color, "size": 16, "symbol": "circle",
                       "line": {"color": "white", "width": 2}},
            "hovertemplate": (f"<b>{animal}</b><br>"
                              "X: %{x:.0f} mm<br>Y: %{y:.0f} mm<br>Z: %{z:.0f} mm"
                              "<extra></extra>"),
        })
        animal_trace_indices[animal] = 3 + i   # 0=room, 1=cameras, 2+=animals

    # ── Compute image display dimensions ──────────────────────────────────────
    native_w, native_h = 2048, 1496
    disp_w = int(native_w * IMG_SCALE)
    disp_h = int(native_h * IMG_SCALE)

    # ── Write HTML ────────────────────────────────────────────────────────────
    print(f"Writing HTML → {OUTPUT_HTML}")

    # JSON-encode large data blobs
    j_frames      = json.dumps(shared_frames)
    j_images      = json.dumps(images)
    j_detections  = json.dumps(det_js)
    j_positions3d = json.dumps({str(k): v for k, v in positions3d.items()})
    j_traces      = json.dumps(plotly_traces)
    j_animal_indices = json.dumps(animal_trace_indices)
    j_animal_colors  = json.dumps(ANIMAL_COLORS)
    j_img_dims    = json.dumps({"native_w": native_w, "native_h": native_h,
                                "disp_w": disp_w, "disp_h": disp_h})

    rx, ry, rz = room["x"], room["y"], room["z"]
    j_layout = json.dumps({
        "paper_bgcolor": "#0a0a1a",
        "font": {"color": "white", "family": "monospace"},
        "scene": {
            "bgcolor": "#0d0d22",
            "xaxis": {"title": "X (mm)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a",
                      "range": [0, rx]},
            "yaxis": {"title": "Y (mm)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a",
                      "range": [0, ry]},
            "zaxis": {"title": "Z (mm)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a",
                      "range": [0, rz]},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 0.8}},
            "aspectmode": "manual",
            "aspectratio": {"x": rx/3400, "y": ry/3400, "z": rz/3400},
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
        "legend": {
            "bgcolor": "rgba(20,20,50,0.85)",
            "bordercolor": "#444477",
            "borderwidth": 1,
            "font": {"color": "white"},
        },
        "uirevision": "keep",
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Datalus — Synchronized Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  display: flex; flex-direction: column; height: 100vh;
  background: #0a0a1a; color: #ccc; font-family: monospace; overflow: hidden;
}}
#header {{
  padding: 8px 16px; background: #0d0d2e; border-bottom: 1px solid #223;
  font-size: 14px; color: #8899bb; flex-shrink: 0;
}}
#main {{
  display: flex; flex: 1; overflow: hidden;
}}
#cam-panel {{
  width: 50%; display: grid; grid-template-columns: 1fr 1fr;
  grid-template-rows: 1fr 1fr; gap: 2px; background: #050510; padding: 2px;
}}
.cam-wrap {{
  position: relative; background: #080818; overflow: hidden;
}}
.cam-wrap canvas {{
  width: 100%; height: 100%; object-fit: contain; display: block;
}}
.cam-label {{
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.65); color: #aabbdd;
  font-size: 11px; padding: 2px 6px; border-radius: 3px; pointer-events: none;
}}
#plot-panel {{
  flex: 1; position: relative;
}}
#plot3d {{
  width: 100%; height: 100%;
}}
#controls {{
  flex-shrink: 0; background: #0d0d2e;
  border-top: 1px solid #223; padding: 10px 16px;
}}
#ctrl-row {{
  display: flex; align-items: center; gap: 12px;
}}
.btn {{
  background: #1a2a4a; border: 1px solid #335; color: #aaccff;
  padding: 5px 14px; border-radius: 4px; cursor: pointer; font-size: 13px;
  font-family: monospace;
}}
.btn:hover {{ background: #243560; }}
#slider {{
  flex: 1; accent-color: #4477cc; height: 4px; cursor: pointer;
}}
#frame-label {{
  min-width: 120px; text-align: right; font-size: 12px; color: #7799bb;
}}
</style>
</head>
<body>
<div id="header">
  Datalus — Synchronized Viewer &nbsp;|&nbsp; session 250711 &nbsp;|&nbsp;
  <span style="color:#3399ff">■ Elm</span> &nbsp;
  <span style="color:#ff5500">■ Jok</span>
</div>
<div id="main">
  <div id="cam-panel">
    <div class="cam-wrap"><canvas id="c102"></canvas><div class="cam-label">cam 102</div></div>
    <div class="cam-wrap"><canvas id="c108"></canvas><div class="cam-label">cam 108</div></div>
    <div class="cam-wrap"><canvas id="c113"></canvas><div class="cam-label">cam 113</div></div>
    <div class="cam-wrap"><canvas id="c117"></canvas><div class="cam-label">cam 117</div></div>
  </div>
  <div id="plot-panel">
    <div id="plot3d"></div>
  </div>
</div>
<div id="controls">
  <div id="ctrl-row">
    <button class="btn" id="btnPlay">&#9654; Play</button>
    <button class="btn" id="btnPause">&#9646;&#9646; Pause</button>
    <input type="range" id="slider" min="0" max="{len(shared_frames)-1}" value="{len(shared_frames)-1}">
    <span id="frame-label">frame {shared_frames[-1]}</span>
  </div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const FRAMES      = {j_frames};
const IMAGES      = {j_images};
const DETECTIONS  = {j_detections};
const POSITIONS3D = {j_positions3d};
const IMG_DIMS    = {j_img_dims};
const ANIMAL_COLORS = {j_animal_colors};
const ANIMAL_TRACE_IDX = {j_animal_indices};
const CAM_IDS     = {json.dumps(CAMERA_IDS)};

// ── Plotly init ────────────────────────────────────────────────────────────────
const traces = {j_traces};
const layout = {j_layout};
Plotly.newPlot('plot3d', traces, layout, {{
  scrollZoom: true, displayModeBar: true,
  modeBarButtonsToRemove: ['toImage'],
  displaylogo: false,
}});

// ── Canvas state ──────────────────────────────────────────────────────────────
const canvases = {{
  '102': document.getElementById('c102'),
  '108': document.getElementById('c108'),
  '113': document.getElementById('c113'),
  '117': document.getElementById('c117'),
}};

// Pre-load Image objects for all frames × cameras (lazy on demand)
const imgCache = {{}};

function getImg(camId, frameIdx) {{
  const key = camId + '_' + frameIdx;
  if (!imgCache[key]) {{
    const im = new Image();
    im.src = IMAGES[camId][frameIdx];
    imgCache[key] = im;
  }}
  return imgCache[key];
}}

// Pre-load the first and nearby frames immediately
(function preload() {{
  for (let i = 0; i < Math.min(5, FRAMES.length); i++) {{
    CAM_IDS.forEach(c => getImg(c, i));
  }}
}})();

function drawCanvas(camId, frameIdx) {{
  const canvas = canvases[camId];
  const ctx = canvas.getContext('2d');
  const dw = canvas.offsetWidth  || IMG_DIMS.disp_w;
  const dh = canvas.offsetHeight || IMG_DIMS.disp_h;
  canvas.width  = dw;
  canvas.height = dh;

  const img = getImg(camId, frameIdx);
  const scaleX = dw / IMG_DIMS.native_w;
  const scaleY = dh / IMG_DIMS.native_h;

  function draw() {{
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(img, 0, 0, dw, dh);

    const frame = FRAMES[frameIdx];
    const dets  = DETECTIONS[frame] || {{}};

    for (const [monkey, camDets] of Object.entries(dets)) {{
      const bbox = camDets[camId];
      if (!bbox) continue;
      const color = ANIMAL_COLORS[monkey] || '#ffffff';
      const x1 = bbox[0]*scaleX, y1 = bbox[1]*scaleY;
      const x2 = bbox[2]*scaleX, y2 = bbox[3]*scaleY;
      ctx.strokeStyle = color;
      ctx.lineWidth   = Math.max(2, dw * 0.003);
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
      ctx.fillStyle = color;
      ctx.font = Math.round(dw * 0.035) + 'px monospace';
      ctx.fillText(monkey, x1 + 3, Math.max(y1 - 4, 14));
    }}
  }}

  if (img.complete) {{
    draw();
  }} else {{
    img.onload = draw;
  }}
}}

function update3D(frameIdx) {{
  const frame = FRAMES[frameIdx].toString();
  const pos   = POSITIONS3D[frame] || {{}};
  const updates = {{ x: [], y: [], z: [] }};
  const traceNums = [];
  for (const [animal, tIdx] of Object.entries(ANIMAL_TRACE_IDX)) {{
    const p = pos[animal];
    updates.x.push(p ? [p[0]] : [null]);
    updates.y.push(p ? [p[1]] : [null]);
    updates.z.push(p ? [p[2]] : [null]);
    traceNums.push(tIdx);
  }}
  Plotly.restyle('plot3d', updates, traceNums);
}}

function setFrame(idx) {{
  CAM_IDS.forEach(c => drawCanvas(c, idx));
  update3D(idx);
  document.getElementById('frame-label').textContent =
    'frame ' + FRAMES[idx] + ' (' + (idx+1) + '/' + FRAMES.length + ')';
  document.getElementById('slider').value = idx;
  // Preload next few frames
  for (let i = idx+1; i < Math.min(idx+4, FRAMES.length); i++) {{
    CAM_IDS.forEach(c => getImg(c, i));
  }}
}}

// Slider
const slider = document.getElementById('slider');
slider.addEventListener('input', () => setFrame(+slider.value));

// Play / Pause
let playTimer = null;
let playIdx   = +slider.value;

document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playTimer) return;
  if (playIdx >= FRAMES.length - 1) playIdx = 0;
  playTimer = setInterval(() => {{
    playIdx++;
    setFrame(playIdx);
    if (playIdx >= FRAMES.length - 1) {{
      clearInterval(playTimer); playTimer = null;
    }}
  }}, 80);
}});

document.getElementById('btnPause').addEventListener('click', () => {{
  clearInterval(playTimer); playTimer = null;
  playIdx = +slider.value;
}});

// Initial render
setFrame(+slider.value);

// Redraw canvases on window resize
window.addEventListener('resize', () => setFrame(+slider.value));
</script>
</body>
</html>
"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    size_mb = OUTPUT_HTML.stat().st_size / 1e6
    print(f"Done. File size: {size_mb:.1f} MB")
    print(f"\nOpening → {OUTPUT_HTML}")
    import os
    os.system(f"open '{OUTPUT_HTML}'")


if __name__ == "__main__":
    main()
