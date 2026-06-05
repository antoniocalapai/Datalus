#!/usr/bin/env python3
"""
HomeCage — Room Viewer
Generates a standalone 3D HTML preview of the room geometry and camera positions.
Auto-discovers all *_result.json files in the output directory and shows each as a
toggleable overlay.

Usage:
    python3 room_viewer.py                          # uses default output/ dir, opens result
    python3 room_viewer.py --no-open                # generate only
    python3 room_viewer.py --dir /path/to/folder    # load JSONs from a custom directory
    python3 room_viewer.py --dir /Volumes/server/output --no-open
"""

import sys, json, subprocess, argparse, base64
from pathlib import Path

# ─── Known geometry ───────────────────────────────────────────────────────────
ROOM = {"x": 2240, "y": 3400, "z": 3260}   # mm

CAMERAS = {
    "102": {"pos": [300,  3260, 540],  "note": "reference — low, far wall"},
    "108": {"pos": [1850, 0,    2480], "note": "high, near wall"},
    "113": {"pos": [50,   0,    550],  "note": "low, near wall"},
    "117": {"pos": [2080, 3070, 2550], "note": "high, far wall"},
}

CAM_COLORS = {"102": "0xff6644", "108": "0xff9933", "113": "0xffcc44", "117": "0xff4477"}
CAM_HEX    = {"102": "ff6644",   "108": "ff9933",   "113": "ffcc44",   "117": "ff4477"}

_DEFAULT_OUT = Path(__file__).parent.parent / "HomeCage_SelfCalibration_Human" / "output"

# ─── Colour palette ───────────────────────────────────────────────────────────
# Keyed by filename stem or label; fallback rotates through PALETTE.
_LABEL_MAP = {
    "calibration_result": "Human Pose",
}

_STYLE_MAP = {
    "calibration_result":        {"hex": "00ddff", "js": "0x00ddff", "line": "0xffff00"},
    "checkerboard_result":       {"hex": "ffaa66", "js": "0xffaa66", "line": "0xff88cc"},
    "monkey_elmo_result":        {"hex": "44aaff", "js": "0x44aaff", "line": "0xffff44"},
    "monkey_joker_result":       {"hex": "ff9144", "js": "0xff9144", "line": "0xff44ff"},
    "monkey_combined_result":    {"hex": "44ff88", "js": "0x44ff88", "line": "0x88ff44"},
    "monkey_ideal_result":       {"hex": "ff44bb", "js": "0xff44bb", "line": "0xffbb44"},
}

_PALETTE = [
    {"hex": "aaffaa", "js": "0xaaffaa", "line": "0x88cc88"},
    {"hex": "ffaaff", "js": "0xffaaff", "line": "0xcc88cc"},
    {"hex": "aaaaff", "js": "0xaaaaff", "line": "0x8888cc"},
    {"hex": "ffff88", "js": "0xffff88", "line": "0xcccc66"},
]


def _style_for(stem, idx):
    if stem in _STYLE_MAP:
        return _STYLE_MAP[stem]
    return _PALETTE[idx % len(_PALETTE)]


# ─── Discover and load result JSONs ───────────────────────────────────────────
def discover_results(out_dir):
    """Return list of (stem, data) for every *_result.json in out_dir."""
    results = []
    skip = {"checkerboard_result", "monkey_elmo_result", "monkey_joker_result",
            "monkey_combined_result", "monkey_ideal_result"}
    for p in sorted(out_dir.glob("*_result.json")):
        if p.stem in skip:
            continue
        try:
            with open(p) as f:
                data = json.load(f)
            results.append((p.stem, data))
        except Exception as e:
            print(f"  Warning: could not load {p.name}: {e}")
    return results


# ─── JS data block for one calibration ───────────────────────────────────────
def calib_js_block(stem, calib, var_name, style):
    entries = []
    for cam_id, info in calib.get("cameras", {}).items():
        if not info.get("placed"):
            continue
        if "reconstructed_aligned_mm" not in info:
            continue
        r = [round(float(v), 0) for v in info["reconstructed_aligned_mm"]]
        k = [round(float(v), 0) for v in info["known_mm"]]
        e = round(float(info["error_mm"]), 0)
        # Forward axis in world: world dir of cam +Z = R_world^T @ [0,0,1]
        fwd = "null"
        if "R_world" in info:
            import numpy as _np
            Rw = _np.array(info["R_world"])
            f = (Rw.T @ _np.array([0, 0, 1.0])).tolist()
            fwd = f"[{f[0]:.4f},{f[1]:.4f},{f[2]:.4f}]"
        entries.append(
            f'  {{id:"{cam_id}",'
            f'recon:[{r[0]},{r[1]},{r[2]}],'
            f'known:[{k[0]},{k[1]},{k[2]}],'
            f'err:{e},fwd:{fwd}}}')

    # Build label and detail line
    label = calib.get("label") or _stem_to_label(stem)

    def _r(v, n=1):
        try:    return round(float(v), n)
        except: return v

    method = calib.get("method", "")
    if method in ("human_pose", "") and "person_height_mm" in calib:
        anchor = "+cam".join(calib.get("anchor_pair", []))
        detail = (f"height {calib.get('person_height_mm','?')}mm · "
                  f"anchor cam{anchor} · "
                  f"reproj {_r(calib.get('reproj_after_ba'),2)}px")
        mean_e = _r(calib.get("mean_error_mm"))
    elif method == "checkerboard":
        pair = "+cam".join(calib.get("stereo_pair", []))
        detail = (f"cam{pair} · {calib.get('n_simultaneous_frames','?')} frames · "
                  f"reproj {_r(calib.get('stereo_reproj_px'),2)}px")
        mean_e = _r(calib.get("distance_error_mm"))
    elif method in ("monkey_pose", "monkey_pose_ideal"):
        anchor = "cam" + "+cam".join(calib.get("anchor_pair", []))
        detail = f"{anchor} · reproj {_r(calib.get('reproj_after_ba'),2)}px"
        mean_e = _r(calib.get("mean_error_mm"))
    else:
        detail = label
        mean_e = _r(calib.get("mean_error_mm"))
    if mean_e is None:
        mean_e = "null"

    lines = ",\n".join(entries)
    return (
        f"const {var_name} = {{\n"
        f'  label: "{label}",\n'
        f'  detail: "{detail}",\n'
        f"  mean_error_mm: {mean_e},\n"
        f"  markerColor: {style['js']},\n"
        f"  lineColor: {style['line']},\n"
        f'  hex: "{style["hex"]}",\n'
        f"  cameras: [\n{lines}\n  ]\n}};"
    )


def _stem_to_label(stem):
    """Convert filename stem to a human-readable label."""
    if stem in _LABEL_MAP:
        return _LABEL_MAP[stem]
    return (stem
            .replace("_result", "")
            .replace("_", " ")
            .title()
            .replace("Monkey ", ""))


# ─── Build HTML ───────────────────────────────────────────────────────────────
def build_html(out_dir):
    results = discover_results(out_dir)

    rx, ry, rz = ROOM["x"], ROOM["y"], ROOM["z"]

    # Optional human track
    track_path = out_dir / "human_track.json"
    if track_path.exists():
        track_js = track_path.read_text()
    else:
        track_js = '{"frames":[]}'

    # Exemplar camera frames (one per camera, from session 250708)
    cam_frames_html = ""
    frames_root = out_dir / "frames" / "250708"
    for cid in CAMERAS:
        cam_dir = frames_root / cid
        if not cam_dir.exists():
            continue
        pngs = sorted(cam_dir.glob("*.png"))
        if not pngs:
            continue
        pick = pngs[len(pngs) // 2]
        b64 = base64.b64encode(pick.read_bytes()).decode("ascii")
        hex_color = CAM_HEX[cid]
        cam_frames_html += (
            f'<div style="margin-bottom:6px">'
            f'<div style="font:bold 11px monospace;color:#{hex_color};margin-bottom:2px">'
            f'Cam {cid}</div>'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;display:block;border:1px solid #{hex_color};border-radius:3px">'
            f'</div>'
        )

    # Known camera JS
    cam_js = ""
    for cid, info in CAMERAS.items():
        px, py, pz = info["pos"]
        cam_js += (f"\n  {{id:'{cid}',pos:[{px},{py},{pz}],"
                   f"note:'{info['note']}',"
                   f"color:{CAM_COLORS[cid]},hex:'#{CAM_HEX[cid]}'}},")

    # Per-calibration blocks
    js_blocks, var_names = [], []
    default_on_flags = []
    for idx, (stem, data) in enumerate(results):
        style   = _style_for(stem, idx)
        vname   = f"CALIB_{idx}"
        block   = calib_js_block(stem, data, vname, style)
        js_blocks.append(block)
        var_names.append(vname)

    all_js      = "\n\n".join(js_blocks) if js_blocks else "// no calibration files found"
    calibs_arr  = "[" + ",".join(var_names) + "]"
    default_on_arr = "[" + ",".join("true" if v else "false" for v in default_on_flags) + "]"

    # Toggle buttons
    DEFAULT_ON = {"calibration_result"}  # plus any monkey_gt_*
    btns = ('<button class="tog on" id="t-known" '
            'style="border-color:#ff8866;color:#ff8866">Known Positions</button>\n  ')
    for idx, (stem, data) in enumerate(results):
        style = _style_for(stem, idx)
        label = data.get("label") or _stem_to_label(stem)
        on = stem in DEFAULT_ON or stem.startswith("monkey_gt_")
        default_on_flags.append(on)
        cls = "tog on" if on else "tog off"
        btns += (f'<button class="{cls}" id="t-{idx}" '
                 f'style="border-color:#{style["hex"]};color:#{style["hex"]}">'
                 f'{label}</button>\n  ')

    # Legend
    legend_lines = []
    for idx, (stem, data) in enumerate(results):
        style = _style_for(stem, idx)
        label = data.get("label") or _stem_to_label(stem)
        legend_lines.append(
            f'<span style="color:#{style["hex"]}">&#9632;</span> {label}')
    legend = "<br>\n  ".join(legend_lines) if legend_lines else "(no calibrations found)"

    n_found = len(results)

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>HomeCage — Room Preview</title>
<style>
  body {{ margin:0; background:#111; overflow:hidden; font-family:monospace; }}
  #info {{ position:absolute; top:12px; left:12px; color:#ccc; font-size:12px;
           background:rgba(0,0,0,0.65); padding:10px 14px; border-radius:6px;
           line-height:1.7; max-width:340px; }}
  #info b {{ color:#8cf; }}
  #toggles {{ position:absolute; bottom:16px; left:50%; transform:translateX(-50%);
              display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }}
  .tog {{ padding:7px 18px; border-radius:20px; border:2px solid; cursor:pointer;
          font:bold 12px monospace; background:rgba(0,0,0,0.7); transition:all .15s; }}
  .tog.on  {{ background:rgba(255,255,255,0.15); }}
  .tog.off {{ opacity:0.45; }}
  .vtog {{ padding:5px 12px; border-radius:14px; border:2px solid; cursor:pointer;
           font:bold 11px monospace; background:rgba(0,0,0,0.7); transition:all .15s;
           text-align:left; }}
  .vtog.on  {{ background:rgba(255,255,255,0.15); }}
  .vtog.off {{ opacity:0.45; }}
</style>
</head>
<body>
<div id="cams" style="position:absolute;top:12px;right:12px;width:220px;
     background:rgba(0,0,0,0.65);padding:8px 10px;border-radius:6px;
     max-height:calc(100vh - 24px);overflow-y:auto;">
  <div style="font:bold 12px monospace;color:#8cf;margin-bottom:6px">
    Camera reference (250708)
  </div>
  {cam_frames_html}
</div>

<div id="viewtoggles" style="position:absolute;top:12px;left:380px;
     background:rgba(0,0,0,0.65);padding:8px 12px;border-radius:6px;
     display:flex;flex-direction:column;gap:6px;">
  <div style="font:bold 11px monospace;color:#8cf;margin-bottom:2px">View</div>
  <button class="vtog on" id="v-human"  style="border-color:#aaffff;color:#aaffff">Human Track</button>
  <button class="vtog on" id="v-room"   style="border-color:#88aacc;color:#88aacc">Room box</button>
  <button class="vtog on" id="v-grid"   style="border-color:#66aa66;color:#66aa66">Floor grid</button>
  <button class="vtog on" id="v-axes"   style="border-color:#cccccc;color:#cccccc">Axes + ticks</button>
  <button class="vtog on" id="v-arrows" style="border-color:#ffaa44;color:#ffaa44">Cam orientation</button>
</div>

<div id="info">
  <b>HomeCage — Room Preview</b><br>
  Room {rx//10}×{ry//10}×{rz//10} cm &nbsp;·&nbsp; drag to orbit, scroll to zoom
  <div id="stats" style="margin-top:6px;border-top:1px solid #333;padding-top:6px;
       font-size:11px;line-height:1.6;"></div>
  <div id="frameinfo" style="margin-top:6px;border-top:1px solid #333;padding-top:6px;
       font-size:11px;color:#aff;"></div>
</div>

<div id="toggles">
  {btns}
</div>
<div id="player" style="position:absolute;bottom:110px;left:50%;transform:translateX(-50%);
     display:flex;align-items:center;gap:10px;background:rgba(0,0,0,0.7);
     padding:6px 14px;border-radius:20px;color:#aaffff;font:11px monospace;">
  <button id="playbtn" style="background:none;border:1px solid #aaffff;color:#aaffff;
          cursor:pointer;padding:3px 10px;border-radius:12px;font:bold 11px monospace;">▶ Play</button>
  <input id="frameslider" type="range" min="0" value="0" step="1" style="width:260px;">
  <span id="framelbl">0/0</span>
</div>

<script type="importmap">
{{ "imports": {{
    "three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"
}} }}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ CSS2DRenderer, CSS2DObject }} from 'three/addons/renderers/CSS2DRenderer.js';

const ROOM    = {{ x:{rx}, y:{ry}, z:{rz} }};
const CAMERAS = [{cam_js}
];

{all_js}

const ALL_CALIBS = {calibs_arr};
const DEFAULT_ON = {default_on_arr};

const R2T = (x,y,z) => new THREE.Vector3(x, z, y);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x141414);

const renderer = new THREE.WebGLRenderer({{ antialias:true }});
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
document.body.appendChild(renderer.domElement);

const labelRenderer = new CSS2DRenderer();
labelRenderer.setSize(innerWidth, innerHeight);
labelRenderer.domElement.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
document.body.appendChild(labelRenderer.domElement);

const cam3 = new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 10, 50000);
cam3.position.set(ROOM.x*0.5, ROOM.z*1.2, -ROOM.y*0.6);

const controls = new OrbitControls(cam3, renderer.domElement);
controls.target.set(ROOM.x/2, ROOM.z/2, ROOM.y/2);
controls.update();

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dl = new THREE.DirectionalLight(0xffffff, 0.8);
dl.position.set(ROOM.x, ROOM.z*2, ROOM.y);
scene.add(dl);

// Room box
const roomGroup = new THREE.Group();
scene.add(roomGroup);
const boxGeo = new THREE.BoxGeometry(ROOM.x, ROOM.z, ROOM.y);
const ctr = new THREE.Vector3(ROOM.x/2, ROOM.z/2, ROOM.y/2);
[new THREE.Mesh(boxGeo, new THREE.MeshBasicMaterial({{ color:0x334455, wireframe:true }})),
 new THREE.Mesh(boxGeo, new THREE.MeshPhongMaterial({{
   color:0x223344, opacity:0.08, transparent:true,
   side:THREE.DoubleSide, depthWrite:false }}))
].forEach(m => {{ m.position.copy(ctr); roomGroup.add(m); }});

const gridGroup = new THREE.Group();
scene.add(gridGroup);
const maxSide = Math.max(ROOM.x, ROOM.y);
const grid = new THREE.GridHelper(maxSide, 20, 0x2a4a2a, 0x1a3a1a);
grid.position.set(ROOM.x/2, 0, ROOM.y/2);
grid.scale.set(ROOM.x/maxSide, 1, ROOM.y/maxSide);
gridGroup.add(grid);

const axesGroup = new THREE.Group();
scene.add(axesGroup);
axesGroup.add(new THREE.AxesHelper(400));
// Axis tick labels in cm along room edges (every 50cm)
const tickLabels = [];
function addTick(html, pos, color) {{
  const d = document.createElement('div');
  d.innerHTML = html;
  d.style.cssText = `color:${{color}};font:10px monospace;`;
  const o = new CSS2DObject(d);
  o.position.copy(pos);
  scene.add(o);
  tickLabels.push(o);
}}
const TICK_MM = 500;
for (let x = 0; x <= ROOM.x; x += TICK_MM)
  addTick(`${{x/10}}cm`, R2T(x, 0, 0), '#f88');
for (let y = 0; y <= ROOM.y; y += TICK_MM)
  addTick(`${{y/10}}cm`, R2T(0, y, 0), '#8f8');
for (let z = 0; z <= ROOM.z; z += TICK_MM)
  addTick(`${{z/10}}cm`, R2T(0, 0, z), '#88f');

function makeLabel(html, pos, color='#aaa') {{
  const d = document.createElement('div');
  d.innerHTML = html;
  d.style.cssText = `color:${{color}};font:11px monospace;`;
  const o = new CSS2DObject(d);
  o.position.copy(pos);
  scene.add(o);
  return o;
}}

// Known camera positions
const knownGroup = new THREE.Group();
const knownLabels = [];
scene.add(knownGroup);
for (const c of CAMERAS) {{
  const [px,py,pz] = c.pos;
  const p = R2T(px,py,pz);
  const mesh = new THREE.Mesh(
    new THREE.OctahedronGeometry(85),
    new THREE.MeshPhongMaterial({{ color:c.color, emissive:c.color, emissiveIntensity:0.3 }}));
  mesh.position.copy(p);
  knownGroup.add(mesh);
  knownGroup.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(p.x,0,p.z), p.clone()]),
    new THREE.LineBasicMaterial({{ color:c.color, opacity:0.4, transparent:true }})));
  const d = document.createElement('div');
  d.innerHTML = `<b>Cam ${{c.id}}</b>`;
  d.style.cssText = `color:#fff;font:12px monospace;background:rgba(0,0,0,0.7);`+
    `padding:4px 8px;border-radius:4px;border-left:3px solid #${{c.hex}};white-space:nowrap;`;
  const lbl = new CSS2DObject(d);
  lbl.position.set(p.x, p.y+130, p.z);
  scene.add(lbl);
  knownLabels.push(lbl);
}}
let knownVisible = true;
const knownBtn = document.getElementById('t-known');
knownBtn.addEventListener('click', () => {{
  knownVisible = !knownVisible;
  knownBtn.className = 'tog ' + (knownVisible ? 'on' : 'off');
  knownVisible ? scene.add(knownGroup) : scene.remove(knownGroup);
  knownLabels.forEach(l => l.visible = knownVisible);
}});

// Overlay builder
function buildOverlay(calib) {{
  const group = new THREE.Group();
  const labels = [];
  for (const c of calib.cameras) {{
    const pr = R2T(...c.recon);
    const pk = R2T(...c.known);
    const mesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(60),
      new THREE.MeshPhongMaterial({{
        color: calib.markerColor, emissive: calib.markerColor,
        emissiveIntensity: 0.35, transparent:true, opacity:0.85
      }}));
    mesh.position.copy(pr);
    group.add(mesh);
    group.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([pr.clone(), pk.clone()]),
      new THREE.LineBasicMaterial({{ color:calib.lineColor, linewidth:2 }})));
    // Orientation arrow (camera forward axis)
    if (c.fwd) {{
      const dir = R2T(c.fwd[0], c.fwd[1], c.fwd[2]);
      const tip = pr.clone().add(dir.multiplyScalar(400));
      const arrowLine = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([pr.clone(), tip]),
        new THREE.LineBasicMaterial({{ color:calib.markerColor }}));
      arrowLine.userData.isArrow = true;
      group.add(arrowLine);
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(40, 100, 8),
        new THREE.MeshBasicMaterial({{ color:calib.markerColor }}));
      cone.position.copy(tip);
      cone.lookAt(pr.clone());
      cone.rotateX(Math.PI / 2);
      cone.userData.isArrow = true;
      group.add(cone);
    }}
    if (c.err > 0) {{
      const mid = pr.clone().lerp(pk, 0.5);
      const d = document.createElement('div');
      d.textContent = `${{c.err.toFixed(0)}}mm`;
      d.style.cssText = `font:bold 11px monospace;background:rgba(0,0,0,0.6);`+
        `padding:1px 5px;border-radius:3px;color:#${{calib.hex}}`;
      const lbl = new CSS2DObject(d);
      lbl.position.copy(mid);
      scene.add(lbl);
      labels.push(lbl);
    }}
  }}
  return {{ group, labels }};
}}

const overlayState = {{}};
for (let i = 0; i < ALL_CALIBS.length; i++) {{
  const calib = ALL_CALIBS[i];
  const ov = buildOverlay(calib);
  const on = DEFAULT_ON[i];
  overlayState[i] = {{ ov, visible:on, calib }};
  if (on) scene.add(ov.group);
  ov.labels.forEach(l => l.visible = on);
}}

function updateStats() {{
  const parts = [];
  for (const [i, s] of Object.entries(overlayState)) {{
    if (!s.visible) continue;
    const c = s.calib;
    const errStr = (!c.mean_error_mm)
      ? ''
      : ` · mean err <b style="color:#ff8">${{c.mean_error_mm}} mm</b>`;
    parts.push(
      `<b style="color:#${{c.hex}}">${{c.label}}</b><br>${{c.detail}}${{errStr}}`);
  }}
  document.getElementById('stats').innerHTML =
    parts.join('<hr style="border-color:#333;margin:3px 0">');
}}
updateStats();

// Toggle buttons
for (let i = 0; i < ALL_CALIBS.length; i++) {{
  const btn = document.getElementById(`t-${{i}}`);
  if (!btn) continue;
  btn.addEventListener('click', () => {{
    const s = overlayState[i];
    s.visible = !s.visible;
    btn.className = 'tog ' + (s.visible ? 'on' : 'off');
    s.visible ? scene.add(s.ov.group) : scene.remove(s.ov.group);
    s.ov.labels.forEach(l => l.visible = s.visible);
    updateStats();
  }});
}}

// ── Human keypoint track ─────────────────────────────────────────────────
const HUMAN_TRACK = {track_js};
const COCO_PAIRS = [
  [0,1],[0,2],[1,3],[2,4],
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16]
];
const humanGroup = new THREE.Group();
scene.add(humanGroup);
const kpMeshes = [];
const kpMat = new THREE.MeshBasicMaterial({{ color:0xaaffff }});
for (let k = 0; k < 17; k++) {{
  const m = new THREE.Mesh(new THREE.SphereGeometry(40, 8, 8), kpMat);
  m.visible = false;
  humanGroup.add(m);
  kpMeshes.push(m);
}}
const boneLines = [];
for (let i = 0; i < COCO_PAIRS.length; i++) {{
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
  const ln = new THREE.Line(g, new THREE.LineBasicMaterial({{ color:0x66ccff }}));
  ln.visible = false;
  humanGroup.add(ln);
  boneLines.push(ln);
}}
let humanVisible = HUMAN_TRACK.frames.length > 0;
let frameIdx = 0;
let lastTick = 0;
const FRAME_DT_MS = 200;  // 5 fps (original video fps)
let humanPlaying = false;

let setHumanFrame = function(i) {{
  if (HUMAN_TRACK.frames.length === 0) return;
  const fr = HUMAN_TRACK.frames[i];
  const kps3d = fr.kps;
  for (let k = 0; k < 17; k++) {{
    const p = kps3d[k];
    if (p) {{
      kpMeshes[k].position.copy(R2T(p[0], p[1], p[2]));
      kpMeshes[k].visible = true;
    }} else {{
      kpMeshes[k].visible = false;
    }}
  }}
  for (let b = 0; b < COCO_PAIRS.length; b++) {{
    const [a, c] = COCO_PAIRS[b];
    const pa = kps3d[a], pc = kps3d[c];
    if (pa && pc) {{
      const arr = boneLines[b].geometry.attributes.position.array;
      const va = R2T(pa[0], pa[1], pa[2]);
      const vc = R2T(pc[0], pc[1], pc[2]);
      arr[0]=va.x; arr[1]=va.y; arr[2]=va.z;
      arr[3]=vc.x; arr[4]=vc.y; arr[5]=vc.z;
      boneLines[b].geometry.attributes.position.needsUpdate = true;
      boneLines[b].visible = true;
    }} else {{
      boneLines[b].visible = false;
    }}
  }}
  document.getElementById('frameinfo').innerHTML =
    `Human track · frame ${{fr.f}} (${{i+1}}/${{HUMAN_TRACK.frames.length}})`;
}}
const slider = document.getElementById('frameslider');
const framelbl = document.getElementById('framelbl');
const playbtn = document.getElementById('playbtn');
slider.max = Math.max(0, HUMAN_TRACK.frames.length - 1);
slider.addEventListener('input', () => {{
  frameIdx = parseInt(slider.value);
  setHumanFrame(frameIdx);
}});
playbtn.addEventListener('click', () => {{
  humanPlaying = !humanPlaying;
  playbtn.textContent = humanPlaying ? '⏸ Pause' : '▶ Play';
}});
const _origSetHumanFrame = setHumanFrame;
setHumanFrame = function(i) {{
  _origSetHumanFrame(i);
  slider.value = i;
  framelbl.textContent = `${{i+1}}/${{HUMAN_TRACK.frames.length}}`;
}};
if (humanVisible) setHumanFrame(0);
else document.getElementById('frameinfo').textContent =
  HUMAN_TRACK.frames.length === 0 ? 'No human_track.json found' : '';

// View toggles
function bindView(id, onToggle) {{
  const b = document.getElementById(id);
  let on = true;
  b.addEventListener('click', () => {{
    on = !on;
    b.className = 'vtog ' + (on ? 'on' : 'off');
    onToggle(on);
  }});
}}
bindView('v-human', v => {{
  humanVisible = v;
  humanGroup.visible = v;
}});
bindView('v-room',  v => {{ roomGroup.visible = v; }});
bindView('v-grid',  v => {{ gridGroup.visible = v; }});
bindView('v-axes',  v => {{
  axesGroup.visible = v;
  tickLabels.forEach(l => l.visible = v);
}});
bindView('v-arrows', v => {{
  // toggle all calibration overlay arrow children (cones + lines after main marker)
  for (const k in overlayState) {{
    overlayState[k].ov.group.children.forEach(ch => {{
      if (ch.userData.isArrow) ch.visible = v;
    }});
  }}
}});

window.addEventListener('resize', () => {{
  cam3.aspect = innerWidth/innerHeight;
  cam3.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  labelRenderer.setSize(innerWidth, innerHeight);
}});

(function animate(t) {{
  requestAnimationFrame(animate);
  if (humanPlaying && humanVisible && HUMAN_TRACK.frames.length > 0 && t - lastTick > FRAME_DT_MS) {{
    lastTick = t;
    frameIdx = (frameIdx + 1) % HUMAN_TRACK.frames.length;
    setHumanFrame(frameIdx);
  }}
  controls.update();
  renderer.render(scene, cam3);
  labelRenderer.render(scene, cam3);
}})();
</script>
</body></html>"""


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HomeCage Room Viewer")
    parser.add_argument("--dir", type=Path, default=_DEFAULT_OUT,
                        help="Directory containing *_result.json files "
                             "(default: HomeCage_SelfCalibration_Human/output)")
    parser.add_argument("--no-open", action="store_true",
                        help="Generate HTML without opening it in the browser")
    args = parser.parse_args()

    out_dir = args.dir.resolve()
    results = discover_results(out_dir)
    print(f"Found {len(results)} calibration file(s) in {out_dir}")
    for stem, data in results:
        label = data.get("label") or _stem_to_label(stem)
        err   = data.get("mean_error_mm")
        print(f"  {stem}: {label}"
              + (f"  mean_err={err}mm" if err else ""))

    out = out_dir / "room_preview.html"
    out.write_text(build_html(out_dir))
    print(f"\nSaved: {out}")

    if not args.no_open:
        subprocess.run(["open", str(out)])
