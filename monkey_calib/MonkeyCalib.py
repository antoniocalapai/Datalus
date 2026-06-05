#!/usr/bin/env python3
"""
MonkeyCalib — Single-animal calibration visualizer  (PyQt6 + pyqtgraph)

Loads two single-monkey sessions (Elmo, Joker) and visualizes their 3D
trajectories using the existing camera calibration.  Toggle between animals
to assess room coverage and calibration quality before running the
monkey-based self-calibration pipeline.

Usage:
    python3 MonkeyCalib.py
    python3 MonkeyCalib.py --calib /path/to/calibration_result.json
"""

import sys, json, re, argparse
import numpy as np
from pathlib import Path

HERE    = Path(__file__).parent
REPO    = HERE.parent
_DEFAULT_CALIB = REPO / "HomeCage_SelfCalibration_Human" / "output" / "calibration_result.json"

MONKEY_ALONE_DIR = REPO / "_data" / "MonkeyAlone"
SESSIONS = {
    "Elmo":  MONKEY_ALONE_DIR / "Elmo _1",
    "Joker": MONKEY_ALONE_DIR / "Joker_1",
}

ANIMAL_NAMES = ["Elmo", "Joker"]

COLORS_F = {
    "Elmo":  (0.27, 0.67, 1.00, 1.0),
    "Joker": (1.00, 0.57, 0.27, 1.0),
}

COLORS_QT = {
    "Elmo":  "#44aaff",
    "Joker": "#ff9144",
}

ROOM       = {"x": 2240, "y": 3400, "z": 3260}
CAM_ORDER  = ["102", "108", "113", "117"]
CAM_KNOWN  = {
    "102": (300,  3260, 540),
    "108": (1850, 0,    2480),
    "113": (50,   0,    550),
    "117": (2080, 3070, 2550),
}

CAM_COLORS_QT = {
    "102": "#ff6644", "108": "#ff9933",
    "113": "#ffcc44", "117": "#ff4477",
}

KP_CONF_THRESH = 0.3
MIN_CAMERAS    = 2
VIDEO_W, VIDEO_H = 2048, 1496

SKEL = [[0,1],[0,2],[1,3],[2,4],
        [5,6],[5,7],[7,9],[6,8],[8,10],
        [5,11],[6,12],[11,12],
        [11,13],[13,15],[12,14],[14,16]]

_CAM_PAT  = re.compile(r'_(\d{3})_\d+.*_2D_result\.(txt|mp4)$')
_TS_PAT   = re.compile(r'_(\d{8})(\d{6})')   # YYYYMMDDHHMMSS in filename

import datetime

def parse_session_start(folder):
    """Return datetime of session start parsed from video/txt filenames, or None."""
    folder = Path(folder)
    for f in sorted(folder.iterdir()):
        m = _TS_PAT.search(f.name)
        if m:
            try:
                return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass
    return None


# ─── Calibration ──────────────────────────────────────────────────────────────
def load_calib(path):
    with open(path) as f:
        return json.load(f)

def build_projection_matrices(calib):
    Ps = {}
    for cid, info in calib["cameras"].items():
        if not info.get("placed") or "R_world" not in info:
            continue
        K = np.array(info["K"])
        R = np.array(info["R_world"])
        T = np.array(info["T_world"]).reshape(3, 1)
        Ps[cid] = K @ np.hstack([R, T])
    return Ps

def cam_world_pos(calib, cid):
    info = calib["cameras"].get(cid, {})
    if "R_world" not in info:
        return None, None
    R = np.array(info["R_world"])
    T = np.array(info["T_world"])
    return -R.T @ T, R

def compute_procrustes(src_pts, dst_pts):
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    sc  = src.mean(0); dc = dst.mean(0)
    sn  = src - sc;    dn = dst - dc
    s   = np.linalg.norm(dn) / np.linalg.norm(sn)
    U, _, Vt = np.linalg.svd(sn.T @ dn)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    return s, R, dc - s * R @ sc

def apply_procrustes(pt, s, R, t):
    if pt is None: return None
    return [round(v, 1) for v in (s * R @ np.array(pt) + t).tolist()]



# ─── 2D results ───────────────────────────────────────────────────────────────
def parse_2d_results(folder):
    folder = Path(folder)
    result = {}
    for txt in sorted(folder.glob("*_2D_result.txt")):
        m = _CAM_PAT.search(txt.name)
        if not m: continue
        cid = m.group(1)
        cam_data = {}
        with open(txt) as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 58: continue
                frame  = int(parts[0])
                bbox   = [float(x) for x in parts[2:7]]
                kps    = [[float(parts[7+i*3]), float(parts[7+i*3+1]),
                           float(parts[7+i*3+2])] for i in range(17)]
                if frame not in cam_data or bbox[4] > cam_data[frame]["bbox"][4]:
                    cam_data[frame] = {"bbox": bbox, "kps": kps}
        result[cid] = cam_data
        print(f"    cam{cid}: {len(cam_data)} frames")
    return result


# ─── Triangulation ────────────────────────────────────────────────────────────
def _triangulate(Ps, pts):
    rows = []
    for P, p in zip(Ps, pts):
        rows.append(p[0]*P[2] - P[0])
        rows.append(p[1]*P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]; return (X[:3] / X[3]).tolist()

def triangulate_session(det2d, Ps):
    all_frames = sorted({f for cd in det2d.values() for f in cd})
    frames_out = {}
    for frame in all_frames:
        cam_Ps_kp  = [[] for _ in range(17)]
        cam_pts_kp = [[] for _ in range(17)]
        bbox_Ps, bbox_pts = [], []
        for cid, cd in det2d.items():
            if cid not in Ps or frame not in cd: continue
            det = cd[frame]
            bbox_Ps.append(Ps[cid])
            b = det["bbox"]
            bbox_pts.append([(b[0]+b[2])/2, (b[1]+b[3])/2])
            for ki, kp in enumerate(det["kps"]):
                if kp[2] >= KP_CONF_THRESH:
                    cam_Ps_kp[ki].append(Ps[cid])
                    cam_pts_kp[ki].append(kp[:2])
        kps_3d = []
        any_valid = False
        for ki in range(17):
            if len(cam_Ps_kp[ki]) >= MIN_CAMERAS:
                kps_3d.append([round(v,1) for v in
                               _triangulate(cam_Ps_kp[ki], cam_pts_kp[ki])])
                any_valid = True
            else:
                kps_3d.append(None)
        if any_valid:
            com = ([round(v,1) for v in _triangulate(bbox_Ps, bbox_pts)]
                   if len(bbox_Ps) >= MIN_CAMERAS else None)
            n_cams = len(bbox_Ps)
            frames_out[frame] = {"kps": kps_3d, "com": com, "n_cams": n_cams}
    return frames_out


# ─── Viewer ───────────────────────────────────────────────────────────────────
def run_viewer(session_data, det2d_all, calib, calib_path):
    """
    session_data: {"Elmo": frames_out_dict, "Joker": frames_out_dict}
    det2d_all:    {"Elmo": det2d_dict,      "Joker": det2d_dict}
    """
    import cv2
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QGridLayout, QLabel, QPushButton, QSlider, QCheckBox,
        QGroupBox, QSizePolicy, QComboBox,
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QKeySequence, QShortcut
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    rx, ry, rz = ROOM["x"], ROOM["y"], ROOM["z"]

    # ── Camera geometry ───────────────────────────────────────────────────────
    cam_pos_physical, cam_pos_recon, cam_R = {}, {}, {}
    for cid, info in calib["cameras"].items():
        if "known_mm" in info:       cam_pos_physical[cid] = list(info["known_mm"])
        if "reconstructed_aligned_mm" in info:
            cam_pos_recon[cid] = list(info["reconstructed_aligned_mm"])
        if "R_world" in info:        cam_R[cid] = np.array(info["R_world"])

    _src = [cam_pos_recon[c]    for c in CAM_ORDER if c in cam_pos_recon and c in cam_pos_physical]
    _dst = [cam_pos_physical[c] for c in CAM_ORDER if c in cam_pos_recon and c in cam_pos_physical]
    proc = compute_procrustes(_src, _dst) if len(_src) >= 3 else None

    frame_mode = [1]   # 0=reconstructed  1=physical

    # ── Per-animal data ───────────────────────────────────────────────────────
    active_animal = ["Elmo"]

    def animal_data():
        return session_data[active_animal[0]]

    def frame_nums_for(name):
        return sorted(session_data[name].keys())

    frame_nums = [frame_nums_for(a) for a in ANIMAL_NAMES]   # [elmo_fns, joker_fns]
    n_frames   = [len(fn) for fn in frame_nums]
    cur_idx    = [0]

    def ai():  # active animal index
        return ANIMAL_NAMES.index(active_animal[0])

    # ── Session stats per animal ──────────────────────────────────────────────
    def compute_stats(name):
        fns = frame_nums_for(name)
        fd_all = session_data[name]
        path_mm = 0.0; prev = None; n_good = 0
        for fn in fns:
            fd = fd_all.get(fn, {})
            com = fd.get("com")
            if com:
                n_good += 1
                if prev: path_mm += float(np.linalg.norm(np.array(com)-np.array(prev)))
                prev = com
        return len(fns), n_good, path_mm

    stats = {a: compute_stats(a) for a in ANIMAL_NAMES}

    # ── Session start times (for absolute time display) ───────────────────────
    session_starts = {a: parse_session_start(SESSIONS[a]) for a in ANIMAL_NAMES}

    FPS = 25.0

    def frame_to_times(frame_num, animal):
        rel_s  = frame_num / FPS
        rm, rs = divmod(int(rel_s), 60)
        rh, rm = divmod(rm, 60)
        rel_str = (f"{rh:02d}:{rm:02d}:{rs:02d}" if rh
                   else f"{rm:02d}:{rs:02d}")
        t0 = session_starts.get(animal)
        if t0:
            abs_dt  = t0 + datetime.timedelta(seconds=rel_s)
            abs_str = abs_dt.strftime("%H:%M:%S")
        else:
            abs_str = "—"
        return rel_str, abs_str

    # ── Videos ───────────────────────────────────────────────────────────────
    def find_videos(folder):
        folder = Path(folder)
        videos = {}
        _cp = re.compile(r'_(\d{3})[_\.]')
        for txt in sorted(folder.glob("*_2D_result.txt")):
            m = _CAM_PAT.search(txt.name)
            if not m: continue
            cid = m.group(1)
            raw = (folder / (txt.stem.replace("_2D_result","") + ".mp4"))
            if raw.exists(): videos[cid] = raw
        if videos: return videos
        for vid in sorted(folder.glob("*.mp4")):
            if "_2D_result" in vid.name: continue
            m = _cp.search(vid.name)
            if m and m.group(1) in CAM_ORDER and m.group(1) not in videos:
                videos[m.group(1)] = vid
        return videos

    all_videos = {a: find_videos(SESSIONS[a]) for a in ANIMAL_NAMES}

    caps = {}
    def open_caps(name):
        for c in caps.values(): c.release()
        caps.clear()
        for cid, vpath in all_videos[name].items():
            cap = cv2.VideoCapture(str(vpath))
            if cap.isOpened(): caps[cid] = cap

    open_caps(active_animal[0])

    def read_frame(cid, frame_num):
        cap = caps.get(cid)
        if cap is None: return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, img = cap.read()
        return img if ok else None

    # ── Window ────────────────────────────────────────────────────────────────
    win = QMainWindow()
    win.setWindowTitle(f"MonkeyCalib · {active_animal[0]} · {calib_path.name}")
    win.resize(1700, 950)

    central = QWidget(); win.setCentralWidget(central)
    root_v  = QVBoxLayout(central)
    root_v.setContentsMargins(4, 4, 4, 4); root_v.setSpacing(2)

    top_row = QWidget()
    root    = QHBoxLayout(top_row)
    root.setContentsMargins(0,0,0,0); root.setSpacing(4)
    root_v.addWidget(top_row, stretch=1)

    # ── Left: GL + stats ─────────────────────────────────────────────────────
    left   = QWidget()
    left_v = QVBoxLayout(left)
    left_v.setContentsMargins(0,0,0,0); left_v.setSpacing(2)

    gl_view = gl.GLViewWidget()
    gl_view.setBackgroundColor((12, 12, 12, 255))
    left_v.addWidget(gl_view)

    info_lbl = QLabel(gl_view)
    info_lbl.setStyleSheet(
        "background:rgba(0,0,0,160);color:#aac8ff;"
        "font:10px monospace;padding:6px 10px;border-radius:6px;")
    info_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    info_lbl.move(8, 8)

    cmp_legend = QLabel(gl_view)
    cmp_legend.setStyleSheet(
        "background:rgba(0,0,0,160);font:9px monospace;padding:5px 8px;border-radius:5px;")
    cmp_legend.setText(
        "<span style='color:#ffffff'>●</span> Known (physical) &nbsp;"
        "<span style='color:#33ccff'>●</span> Human pose reconstruction")
    cmp_legend.adjustSize()
    cmp_legend.move(8, 120)
    cmp_legend.setVisible(False)

    stats_lbl = QLabel()
    stats_lbl.setFixedHeight(20)
    stats_lbl.setStyleSheet(
        "background:#0d1a0d;color:#88cc88;font:9px monospace;padding:2px 6px;")
    stats_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    left_v.addWidget(stats_lbl)

    # ── Playback bar ──────────────────────────────────────────────────────────
    ctrl_bar = QWidget(); ctrl_bar.setFixedHeight(38)
    ctrl_h   = QHBoxLayout(ctrl_bar)
    ctrl_h.setContentsMargins(4,0,4,0); ctrl_h.setSpacing(6)

    btn_play = QPushButton("▶"); btn_play.setFixedSize(28, 34)
    btn_play.setStyleSheet("font-size:11px;padding:0;")

    class FrameTicksBar(QWidget):
        def __init__(self):
            super().__init__(); self.setFixedHeight(8)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        def paintEvent(self, _e):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(18, 18, 18))
            fns = frame_nums[ai()]
            nf  = len(fns)
            if nf <= 1: return
            w = max(1, self.width()-1)
            col = QColor(*[int(c*255) for c in COLORS_F[active_animal[0]][:3]], 180)
            for idx, fn in enumerate(fns):
                fd = session_data[active_animal[0]].get(fn, {})
                if fd.get("com"):
                    x = int(idx / (nf-1) * w)
                    p.fillRect(x, 0, 2, 8, col)
        def mousePressEvent(self, e):
            fns = frame_nums[ai()]; nf = len(fns)
            if nf <= 1: return
            raw = round(e.pos().x() / max(1, self.width()-1) * (nf-1))
            set_frame(max(0, min(nf-1, raw)))

    slider_container = QWidget()
    slider_v = QVBoxLayout(slider_container)
    slider_v.setContentsMargins(0,0,0,0); slider_v.setSpacing(1)
    ticks_bar = FrameTicksBar()
    slider_v.addWidget(ticks_bar)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, max(0, n_frames[0]-1))
    slider.setFixedHeight(18)
    slider_v.addWidget(slider)

    lbl_frame = QLabel("#—")
    lbl_frame.setFixedWidth(210)
    lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl_frame.setStyleSheet("font-size:10px;")

    speed_combo = QComboBox(); speed_combo.setFixedSize(60, 28)
    speed_combo.setStyleSheet("font-size:10px;")
    for ls, val in [("0.25×",.25),("0.5×",.5),("1×",1.0),("2×",2.0),("5×",5.0)]:
        speed_combo.addItem(ls, val)
    speed_combo.setCurrentIndex(2)

    ctrl_h.addWidget(btn_play)
    ctrl_h.addWidget(slider_container, stretch=1)
    ctrl_h.addWidget(lbl_frame)
    ctrl_h.addWidget(speed_combo)

    root.addWidget(left, stretch=3)

    # ── Right panel ───────────────────────────────────────────────────────────
    right   = QWidget(); right.setFixedWidth(660)
    right_v = QVBoxLayout(right)
    right_v.setContentsMargins(0,0,0,0); right_v.setSpacing(4)

    # Animal toggle box
    animal_box = QGroupBox("Animal")
    animal_h   = QHBoxLayout(animal_box)
    animal_h.setSpacing(8)

    animal_btns = {}
    for name in ANIMAL_NAMES:
        btn = QPushButton(f"● {name}")
        btn.setFixedHeight(34)
        btn.setCheckable(True)
        animal_btns[name] = btn
        animal_h.addWidget(btn)
    animal_btns["Elmo"].setChecked(True)

    def _style_animal_btns():
        for name, btn in animal_btns.items():
            c  = COLORS_QT[name]
            bg = "#0d1a2a" if name == "Elmo" else "#2a1a0d"
            if btn.isChecked():
                btn.setStyleSheet(
                    f"font-size:12px;font-weight:bold;padding:4px;"
                    f"background:{bg};color:{c};border:2px solid {c};border-radius:4px;")
            else:
                btn.setStyleSheet(
                    "font-size:12px;padding:4px;"
                    "background:#1a1a1a;color:#555;border:1px solid #333;border-radius:4px;")

    _style_animal_btns()

    # Display toggles box
    tog_box = QGroupBox("Display")
    tog_v   = QVBoxLayout(tog_box)
    tog_v.setSpacing(4)

    def make_cb(label, checked=True):
        cb = QCheckBox(label); cb.setChecked(checked); return cb

    cb_skel    = make_cb("Skeleton")
    cb_dots    = make_cb("Keypoints")
    cb_com     = make_cb("CoM sphere")
    cb_trail   = make_cb("Trail", False)
    cb_videos  = make_cb("Video feeds")
    cb_cam_cmp = make_cb("Camera comparison", False)

    trail_combo = QComboBox()
    for lt, val in [("30 f",30),("60 f",60),("2 min",120),("5 min",300),("∞",99999)]:
        trail_combo.addItem(lt, val)
    trail_combo.setCurrentIndex(1)
    trail_row = QWidget(); tr_h = QHBoxLayout(trail_row)
    tr_h.setContentsMargins(0,0,0,0)
    tr_h.addWidget(cb_trail); tr_h.addWidget(trail_combo); tr_h.addStretch()

    mode_combo = QComboBox()
    mode_combo.addItem("Mode A — Reconstructed", 0)
    mode_combo.addItem("Mode B — Physical  (Procrustes)", 1)
    mode_combo.setCurrentIndex(1); mode_combo.setStyleSheet("font-size:10px;")

    for w in [mode_combo, cb_skel, cb_dots, cb_com, trail_row, cb_videos, cb_cam_cmp]:
        tog_v.addWidget(w)

    # Coverage stats box
    cov_box = QGroupBox("Coverage")
    cov_v   = QVBoxLayout(cov_box)
    cov_v.setSpacing(4)

    cov_labels = {}
    for name in ANIMAL_NAMES:
        nf, ng, pm = stats[name]
        c   = COLORS_QT[name]
        lbl = QLabel(
            f"<b style='color:{c}'>{name}</b>  "
            f"{ng}/{nf} frames tracked  ·  "
            f"path {pm/1000:.1f} m")
        lbl.setStyleSheet("font-size:10px;")
        cov_v.addWidget(lbl)
        cov_labels[name] = lbl

    top_ctrl = QWidget(); top_ctrl_h = QHBoxLayout(top_ctrl)
    top_ctrl_h.setContentsMargins(0,0,0,0); top_ctrl_h.setSpacing(4)
    top_ctrl_h.addWidget(tog_box, stretch=2)
    top_ctrl_h.addWidget(cov_box, stretch=3)

    right_v.addWidget(animal_box)
    right_v.addWidget(top_ctrl)

    # Video grid
    class ClickLabel(QLabel):
        def __init__(self, cid, *a, **kw):
            super().__init__(*a, **kw); self.cid = cid
        def mousePressEvent(self, _): snap_camera(self.cid)

    vid_grid_widget = QWidget()
    vid_grid = QGridLayout(vid_grid_widget)
    vid_grid.setContentsMargins(0,0,0,0); vid_grid.setSpacing(2)

    vid_labels = {}
    for i, cid in enumerate(CAM_ORDER):
        row, col = divmod(i, 2)
        lbl = ClickLabel(cid)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"background:#111; border:2px solid #444;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumSize(200, 150)
        lbl.setText(f"Cam {cid}")
        vid_grid.addWidget(lbl, row, col)
        vid_labels[cid] = lbl

    right_v.addWidget(vid_grid_widget, stretch=1)
    root.addWidget(right)
    root_v.addWidget(ctrl_bar)

    # ── 3D scene ──────────────────────────────────────────────────────────────
    corners = np.array([[x,y,z] for x in [0,rx] for y in [0,ry] for z in [0,rz]])
    for a, b in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
                 (4,5),(4,6),(5,7),(6,7)]:
        gl_view.addItem(gl.GLLinePlotItem(
            pos=np.array([corners[a],corners[b]], dtype=float),
            color=(0.13,0.27,0.4,0.5), width=1, antialias=True))

    grid = gl.GLGridItem()
    grid.setSize(x=rx, y=ry); grid.setSpacing(x=rx/10, y=ry/10)
    grid.translate(rx/2, ry/2, 0); gl_view.addItem(grid)

    # Camera cones
    def make_cone_meshdata(pos, R, radius=60, height=120, n=20):
        pos  = np.array(pos, dtype=float)
        fwd  = np.array(R)[2]; fwd /= max(np.linalg.norm(fwd), 1e-9)
        ref  = np.array([0,0,1.]) if abs(fwd[2]) < 0.9 else np.array([1,0,0.])
        p1   = np.cross(fwd, ref);  p1 /= np.linalg.norm(p1)
        p2   = np.cross(fwd, p1)
        angs = np.linspace(0, 2*np.pi, n, endpoint=False)
        base = np.array([pos + radius*(np.cos(a)*p1 + np.sin(a)*p2) for a in angs])
        tip  = pos + fwd*height
        verts = np.vstack([base, [tip], [pos]])
        faces = []
        for i in range(n):
            j = (i+1)%n
            faces.append([i, j, n]); faces.append([n+1, j, i])
        return gl.MeshData(vertexes=verts, faces=np.array(faces))

    cam_cone_items = {}; active_snap = [None]

    def _cam_pos(cid):
        if frame_mode[0] == 1:
            return cam_pos_physical.get(cid, list(CAM_KNOWN.get(cid, (0,0,0))))
        return cam_pos_recon.get(cid, list(CAM_KNOWN.get(cid, (0,0,0))))

    for cid in CAM_ORDER:
        R  = cam_R.get(cid, np.eye(3))
        md = make_cone_meshdata(_cam_pos(cid), R)
        cone = gl.GLMeshItem(meshdata=md, color=(1.,0.18,0.18,0.88),
                             smooth=True, drawEdges=False, glOptions='translucent')
        gl_view.addItem(cone)
        cam_cone_items[cid] = cone
        try:
            px,py,pz = _cam_pos(cid)
            txt = gl.GLTextItem(pos=np.array([px,py,pz+160],dtype=float),
                                text=f"Cam {cid}", color=(255,80,80,255))
            gl_view.addItem(txt)
        except Exception: pass

    def update_cam_cones():
        for cid in CAM_ORDER:
            R  = cam_R.get(cid, np.eye(3))
            md = make_cone_meshdata(_cam_pos(cid), R)
            cam_cone_items[cid].setMeshData(meshdata=md)

    # ── Skeleton GL objects ───────────────────────────────────────────────────
    col_f   = COLORS_F[active_animal[0]]
    skel_lines = []
    for _ in SKEL:
        ln = gl.GLLinePlotItem(pos=np.zeros((2,3),dtype=float),
                               color=col_f, width=2, antialias=True)
        ln.setVisible(False); gl_view.addItem(ln)
        skel_lines.append(ln)

    dot_item = gl.GLScatterPlotItem(
        pos=np.zeros((1,3),dtype=float), color=np.array([col_f]), size=8, pxMode=True)
    dot_item.setVisible(False); gl_view.addItem(dot_item)

    com_item = gl.GLScatterPlotItem(
        pos=np.zeros((1,3),dtype=float), color=np.array([col_f]), size=250, pxMode=False)
    com_item.setVisible(False); gl_view.addItem(com_item)

    trail_line = gl.GLLinePlotItem(
        pos=np.zeros((2,3),dtype=float), color=(*col_f[:3],0.45), width=2, antialias=True)
    trail_line.setVisible(False); gl_view.addItem(trail_line)
    trail_pts = []

    def _recolor_skeleton(name):
        c = COLORS_F[name]
        for ln in skel_lines:
            ln.setData(color=c)
        dot_item.setData(color=np.array([c]))
        com_item.setData(color=np.array([c]))
        trail_line.setData(color=(*c[:3], 0.45))

    # ── Camera comparison GL items ────────────────────────────────────────────
    # Two marker sets: known physical (white) and HomeCage human reconstruction (cyan)
    _CMP_COLS = {
        "known":    (1.00, 1.00, 1.00, 0.9),
        "homecage": (0.20, 0.80, 1.00, 0.9),
    }

    def _build_cmp_pts(source):
        if source == "known":
            pts = [cam_pos_physical.get(c, CAM_KNOWN.get(c)) for c in CAM_ORDER]
        else:  # homecage
            pts = [cam_pos_recon.get(c) for c in CAM_ORDER]
        return np.array([p for p in pts if p is not None], dtype=float)

    cmp_items = {}
    for src, col in _CMP_COLS.items():
        pts = _build_cmp_pts(src)
        item = gl.GLScatterPlotItem(
            pos=pts if len(pts) else np.zeros((1,3), float),
            color=np.array([col]*max(1,len(pts))),
            size=18, pxMode=True)
        item.setVisible(False)
        gl_view.addItem(item)
        cmp_items[src] = item

    # Connecting lines between known and reconstructed per camera
    cmp_lines = []
    for cid in CAM_ORDER:
        p1 = cam_pos_physical.get(cid, CAM_KNOWN.get(cid))
        p2 = cam_pos_recon.get(cid)
        if p1 and p2:
            ln = gl.GLLinePlotItem(
                pos=np.array([p1, p2], dtype=float),
                color=(0.6, 0.6, 0.6, 0.5), width=1, antialias=True)
            ln.setVisible(False)
            gl_view.addItem(ln)
            cmp_lines.append(ln)

    def _update_cmp_visibility(visible):
        for item in cmp_items.values(): item.setVisible(visible)
        for ln in cmp_lines:           ln.setVisible(visible)

    # ── Camera snap ───────────────────────────────────────────────────────────
    room_center = np.array([rx/2, ry/2, rz/4])

    def snap_camera(cid):
        if frame_mode[0] == 1:
            pos = cam_pos_physical.get(cid)
            if pos is None: return
            C = np.array(pos, dtype=float)
        else:
            C, _ = cam_world_pos(calib, cid)
            if C is None: return
        fc  = C - room_center
        dist = float(np.linalg.norm(fc))
        if dist < 1: return
        d    = fc/dist
        elev = float(np.degrees(np.arcsin(np.clip(d[2],-1,1))))
        az   = float(np.degrees(np.arctan2(d[1], d[0])))
        gl_view.opts['center'] = pg.Vector(*room_center)
        gl_view.setCameraPosition(distance=dist+300, elevation=elev, azimuth=az)
        if active_snap[0] and active_snap[0] != cid:
            cam_cone_items[active_snap[0]].setVisible(True)
            vid_labels[active_snap[0]].setStyleSheet(
                "background:#111; border:2px solid #444;")
        cam_cone_items[cid].setVisible(False)
        vid_labels[cid].setStyleSheet(
            "background:#111; border:3px solid #ff6600;")
        active_snap[0] = cid
        gl_view.update()

    # ── xform ─────────────────────────────────────────────────────────────────
    def _xform(pt):
        if frame_mode[0] == 0 or proc is None: return pt
        return apply_procrustes(pt, *proc)

    # ── set_frame ─────────────────────────────────────────────────────────────
    def set_frame(idx):
        cur_idx[0] = idx
        slider.blockSignals(True); slider.setValue(idx); slider.blockSignals(False)
        name = active_animal[0]
        fns  = frame_nums[ai()]
        if idx >= len(fns): return
        fn   = fns[idx]
        fd   = session_data[name].get(fn, {})
        kps  = fd.get("kps"); com_raw = fd.get("com")
        n_c  = fd.get("n_cams", 0)

        com = _xform(com_raw) if com_raw else None

        # trail
        if com:
            trail_pts.append(com[:])
            tl = trail_combo.currentData()
            if len(trail_pts) > tl: trail_pts[:] = trail_pts[-tl:]

        if len(trail_pts) > 1 and cb_trail.isChecked():
            trail_line.setData(pos=np.array(trail_pts, dtype=float))
            trail_line.setVisible(True)
        else:
            trail_line.setVisible(False)

        if com and cb_com.isChecked():
            com_item.setData(pos=np.array([com], dtype=float))
            com_item.setVisible(True)
        else:
            com_item.setVisible(False)

        if kps:
            valid = [_xform(kp) for kp in kps if kp is not None]
            xkps  = [_xform(kp) for kp in kps]
            if valid and cb_dots.isChecked():
                dot_item.setData(pos=np.array(valid, dtype=float))
                dot_item.setVisible(True)
            else:
                dot_item.setVisible(False)
            for bi, (a, b) in enumerate(SKEL):
                pa, pb = xkps[a], xkps[b]
                if pa and pb and cb_skel.isChecked():
                    skel_lines[bi].setData(pos=np.array([pa,pb], dtype=float))
                    skel_lines[bi].setVisible(True)
                else:
                    skel_lines[bi].setVisible(False)
        else:
            dot_item.setVisible(False)
            for ln in skel_lines: ln.setVisible(False)

        # info overlay
        c = COLORS_QT[name]
        rel_str, abs_str = frame_to_times(fn, name)
        info_lbl.setText(
            f"<b style='color:{c}'>{name}</b><br>"
            f"<b>Frame:</b> #{fn}  ({idx+1}/{len(fns)})"
            f"  <b>+{rel_str}</b>  <b>{abs_str}</b><br>"
            f"<b>Cameras:</b> {n_c}  <b>Mode:</b> "
            f"{'A·Recon' if frame_mode[0]==0 else 'B·Physical'}<br>"
            + (f"<b>CoM:</b> ({com[0]:.0f},{com[1]:.0f},{com[2]:.0f}) mm"
               if com else "<i>no CoM</i>")
        )
        info_lbl.adjustSize()
        lbl_frame.setText(f"+{rel_str}  {abs_str}  #{fn}")

        if cb_videos.isChecked():
            update_videos(fn)

    def update_videos(frame_num):
        name    = active_animal[0]
        col_bgr = tuple(int(c*255) for c in reversed(COLORS_F[name][:3]))
        det2d   = det2d_all.get(name, {})

        for cid in CAM_ORDER:
            lbl = vid_labels[cid]
            w, h = lbl.width(), lbl.height()
            if w < 10 or h < 10: continue
            img = read_frame(cid, frame_num)
            if img is None:
                lbl.setText(f"Cam {cid}\n(no video)"); continue

            # ── draw bbox + keypoints ─────────────────────────────────────
            det = det2d.get(cid, {}).get(frame_num)
            if det is not None:
                bx1,by1,bx2,by2,_ = [int(v) for v in det["bbox"]]
                cv2.rectangle(img, (bx1,by1), (bx2,by2), col_bgr, 2)
                kps = det["kps"]
                # skeleton lines
                for a, b in SKEL:
                    ka, kb = kps[a], kps[b]
                    if ka[2] >= KP_CONF_THRESH and kb[2] >= KP_CONF_THRESH:
                        cv2.line(img, (int(ka[0]),int(ka[1])),
                                 (int(kb[0]),int(kb[1])), col_bgr, 2)
                # keypoint dots
                for kp in kps:
                    if kp[2] >= KP_CONF_THRESH:
                        cv2.circle(img, (int(kp[0]),int(kp[1])), 4,
                                   (255,255,255), -1)
                        cv2.circle(img, (int(kp[0]),int(kp[1])), 4,
                                   col_bgr, 1)

            # ── cam label ─────────────────────────────────────────────────
            cv2.putText(img, f"Cam {cid}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30,30,30), 6)
            cv2.putText(img, f"Cam {cid}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col_bgr, 3)

            scale = min(w/VIDEO_W, h/VIDEO_H)
            nw, nh = max(1,int(VIDEO_W*scale)), max(1,int(VIDEO_H*scale))
            img_rgb = cv2.cvtColor(cv2.resize(img,(nw,nh)), cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, nw, nh,
                          img_rgb.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap(w, h); pix.fill(QColor(17,17,17))
            p = QPainter(pix)
            p.drawPixmap((w-nw)//2, (h-nh)//2, QPixmap.fromImage(qimg))
            p.end()
            lbl.setPixmap(pix)

    # ── update stats bar ──────────────────────────────────────────────────────
    def update_stats_bar():
        name = active_animal[0]
        nf, ng, pm = stats[name]
        c = COLORS_QT[name]
        fns = frame_nums[ai()]
        dur_s = (fns[-1] - fns[0]) / 25.0 if len(fns) > 1 else 0
        tm, ts = divmod(int(dur_s), 60)
        stats_lbl.setText(
            f"<b style='color:{c}'>{name}</b>  "
            f"Tracked: {ng}/{nf} frames  ·  "
            f"Path: {pm/1000:.1f} m  ·  "
            f"Duration: {tm:02d}:{ts:02d}")

    # ── switch animal ─────────────────────────────────────────────────────────
    def switch_animal(name):
        if name == active_animal[0]: return
        active_animal[0] = name
        trail_pts.clear()

        for n, btn in animal_btns.items():
            btn.setChecked(n == name)
        _style_animal_btns()
        _recolor_skeleton(name)

        open_caps(name)

        nf = n_frames[ai()]
        slider.blockSignals(True)
        slider.setRange(0, max(0, nf-1))
        slider.setValue(0)
        slider.blockSignals(False)
        cur_idx[0] = 0
        ticks_bar.update()
        update_stats_bar()
        win.setWindowTitle(f"MonkeyCalib · {name} · {calib_path.name}")
        set_frame(0)

    for name in ANIMAL_NAMES:
        animal_btns[name].clicked.connect(
            lambda _c, n=name: switch_animal(n))

    # ── playback ──────────────────────────────────────────────────────────────
    playing = [False]; timer = QTimer()

    def tick():
        speed = speed_combo.currentData()
        tick.skip = getattr(tick, 'skip', 0)
        if speed < 1:
            tick.skip += 1
            if tick.skip < int(round(1/speed)): return
            tick.skip = 0
        nf = n_frames[ai()]
        set_frame((cur_idx[0] + max(1,int(round(speed)))) % nf)

    timer.timeout.connect(tick)

    def toggle_play():
        playing[0] = not playing[0]
        if playing[0]:
            timer.setInterval(max(40, int(200/speed_combo.currentData())))
            timer.start(); btn_play.setText("⏸")
        else:
            timer.stop(); btn_play.setText("▶")

    btn_play.clicked.connect(toggle_play)
    slider.valueChanged.connect(set_frame)
    speed_combo.currentIndexChanged.connect(
        lambda _: timer.setInterval(max(40, int(200/speed_combo.currentData())))
                  if playing[0] else None)
    cb_videos.stateChanged.connect(
        lambda _: (vid_grid_widget.setVisible(cb_videos.isChecked()),
                   update_videos(frame_nums[ai()][cur_idx[0]])
                   if cb_videos.isChecked() else None))

    def on_mode_change(_):
        frame_mode[0] = mode_combo.currentData()
        trail_pts.clear()
        update_cam_cones()
        set_frame(cur_idx[0])

    mode_combo.currentIndexChanged.connect(on_mode_change)
    for cb in [cb_skel, cb_dots, cb_com, cb_trail]:
        cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))
    trail_combo.currentIndexChanged.connect(lambda _: set_frame(cur_idx[0]))
    def _on_cmp_toggle(_):
        v = cb_cam_cmp.isChecked()
        _update_cmp_visibility(v)
        cmp_legend.setVisible(v)
    cb_cam_cmp.stateChanged.connect(_on_cmp_toggle)

    # ── keyboard shortcuts ────────────────────────────────────────────────────
    QShortcut(QKeySequence(Qt.Key.Key_Space), win).activated.connect(toggle_play)
    QShortcut(QKeySequence(Qt.Key.Key_E),     win).activated.connect(
        lambda: switch_animal("Elmo"))
    QShortcut(QKeySequence(Qt.Key.Key_J),     win).activated.connect(
        lambda: switch_animal("Joker"))
    QShortcut(QKeySequence(Qt.Key.Key_1), win).activated.connect(
        lambda: snap_camera(CAM_ORDER[0]))
    QShortcut(QKeySequence(Qt.Key.Key_2), win).activated.connect(
        lambda: snap_camera(CAM_ORDER[1]))
    QShortcut(QKeySequence(Qt.Key.Key_3), win).activated.connect(
        lambda: snap_camera(CAM_ORDER[2]))
    QShortcut(QKeySequence(Qt.Key.Key_4), win).activated.connect(
        lambda: snap_camera(CAM_ORDER[3]))

    # ── Show ──────────────────────────────────────────────────────────────────
    win.show()
    gl_view.opts['center'] = pg.Vector(rx/2, ry/2, rz/4)
    gl_view.setCameraPosition(distance=max(rx,ry,rz)*3.0, elevation=30, azimuth=45)
    update_stats_bar()
    set_frame(0)
    sys.exit(app.exec())


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MonkeyCalib visualizer")
    ap.add_argument("--calib", default=None,
                    help="Path to calibration_result.json (auto-detected if omitted)")
    args = ap.parse_args()

    calib_path = Path(args.calib) if args.calib else _DEFAULT_CALIB
    if not calib_path.exists():
        print(f"ERROR: calibration not found: {calib_path}"); sys.exit(1)

    print(f"Calibration: {calib_path}")
    calib = load_calib(calib_path)
    Ps    = build_projection_matrices(calib)
    print(f"  Cameras: {sorted(Ps.keys())}")

    session_data = {}
    det2d_all    = {}
    for name, folder in SESSIONS.items():
        if not folder.exists():
            print(f"WARNING: {name} session folder not found: {folder}")
            session_data[name] = {}; det2d_all[name] = {}
            continue
        print(f"\n── {name} ({folder.name}) ──")
        det2d = parse_2d_results(folder)
        det2d_all[name] = det2d
        print(f"  Triangulating…")
        session_data[name] = triangulate_session(det2d, Ps)
        print(f"  Frames with 3D output: {len(session_data[name])}")

    print("\n── Launching viewer ──")
    run_viewer(session_data, det2d_all, calib, calib_path)
