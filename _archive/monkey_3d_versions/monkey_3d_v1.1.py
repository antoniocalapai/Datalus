#!/usr/bin/env python3
"""
HomeCage — Monkey 3D Triangulation & Viewer  (PyQt6 + pyqtgraph + cv2)

Usage:
    python3 monkey_3d.py <2d_results_folder>
"""

import sys, json, re, datetime
import numpy as np
from pathlib import Path

HERE    = Path(__file__).parent
CALIB_F = HERE.parent / "output" / "calibration_result.json"

KP_CONF_THRESH = 0.3
MIN_CAMERAS    = 2
VIDEO_W        = 2048
VIDEO_H        = 1496

ROOM = {"x": 2240, "y": 3400, "z": 3260}

SKEL = [[0,1],[0,2],[1,3],[2,4],
        [5,6],[5,7],[7,9],[6,8],[8,10],
        [5,11],[6,12],[11,12],
        [11,13],[13,15],[12,14],[14,16]]

MONKEY_COLORS_F = [
    (0.27, 0.67, 1.00, 1.0),
    (1.00, 0.47, 0.27, 1.0),
    (0.27, 1.00, 0.53, 1.0),
    (1.00, 0.87, 0.27, 1.0),
]
MONKEY_COLORS_QT = ["#44aaff", "#ff7744", "#44ff88", "#ffdd44"]

CAM_ORDER = ["102", "108", "113", "117"]
CAM_KNOWN_POS = {
    "102": (300,  3260, 540),
    "108": (1850, 0,    2480),
    "113": (50,   0,    550),
    "117": (2080, 3070, 2550),
}
CAM_COLORS_F = {
    "102": (1.0, 0.4, 0.27, 1.0),
    "108": (1.0, 0.6, 0.2,  1.0),
    "113": (1.0, 0.8, 0.27, 1.0),
    "117": (1.0, 0.27,0.47, 1.0),
}
CAM_COLORS_QT = {
    "102": "#ff6644", "108": "#ff9933",
    "113": "#ffcc44", "117": "#ff4477",
}

_CAM_PAT = re.compile(r'_(\d{3})_\d+.*_2D_result\.(txt|mp4)$')


# ─── Calibration ──────────────────────────────────────────────────────────────
def load_calib():
    with open(CALIB_F) as f:
        return json.load(f)

def build_projection_matrices(calib):
    Ps = {}
    for cam_id, info in calib["cameras"].items():
        if not info.get("placed") or "R_world" not in info:
            continue
        K = np.array(info["K"])
        R = np.array(info["R_world"])
        T = np.array(info["T_world"]).reshape(3, 1)
        Ps[cam_id] = K @ np.hstack([R, T])
    return Ps

def cam_world_pose(calib, cid):
    info = calib["cameras"].get(cid, {})
    if "R_world" not in info:
        return None, None
    R = np.array(info["R_world"])
    T = np.array(info["T_world"])
    return -R.T @ T, R


def compute_procrustes(src_pts, dst_pts):
    src = np.array(src_pts, dtype=float)
    dst = np.array(dst_pts, dtype=float)
    src_c = src.mean(0); dst_c = dst.mean(0)
    src_n = src - src_c;  dst_n = dst - dst_c
    s = np.linalg.norm(dst_n) / np.linalg.norm(src_n)
    U, _, Vt = np.linalg.svd(src_n.T @ dst_n)
    R_proc = Vt.T @ U.T
    if np.linalg.det(R_proc) < 0:
        Vt[-1] *= -1
        R_proc = Vt.T @ U.T
    return s, R_proc, dst_c - s * R_proc @ src_c


def apply_procrustes(pt, s, R_proc, t_proc):
    if pt is None:
        return None
    return [round(v, 1) for v in (s * R_proc @ np.array(pt) + t_proc).tolist()]


# ─── Parse 2D results ─────────────────────────────────────────────────────────
def parse_2d_results(folder):
    folder = Path(folder)
    result = {}
    for txt in sorted(folder.glob("*_2D_result.txt")):
        m = _CAM_PAT.search(txt.name)
        if not m:
            continue
        cam_id = m.group(1)
        cam_data = {}
        n = 0
        with open(txt) as f:
            next(f)
            for line in f:
                parts = line.split()
                if len(parts) < 58:
                    continue
                frame  = int(parts[0])
                monkey = parts[1]
                bbox   = [float(x) for x in parts[2:7]]
                kps    = [[float(parts[7+i*3]), float(parts[7+i*3+1]),
                           float(parts[7+i*3+2])] for i in range(17)]
                if frame not in cam_data:
                    cam_data[frame] = {}
                if monkey not in cam_data[frame] or \
                   bbox[4] > cam_data[frame][monkey]["bbox"][4]:
                    cam_data[frame][monkey] = {"bbox": bbox, "kps": kps}
                n += 1
        result[cam_id] = cam_data
        print(f"  cam{cam_id}: {n} detections across {len(cam_data)} frames")
    return result

def find_videos(folder):
    folder = Path(folder)
    videos = {}
    for vid in sorted(folder.glob("*_2D_result.mp4")):
        m = _CAM_PAT.search(vid.name)
        if m:
            videos[m.group(1)] = vid
    return videos


# ─── Triangulation ────────────────────────────────────────────────────────────
def _triangulate(Ps, pts):
    rows = []
    for P, p in zip(Ps, pts):
        rows.append(p[0]*P[2] - P[0])
        rows.append(p[1]*P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return (X[:3] / X[3]).tolist()

def triangulate_session(det2d, Ps):
    all_frames = sorted({f for cd in det2d.values() for f in cd})
    print(f"  Union of frames with detections: {len(all_frames)}")
    frames_out = {}
    for frame in all_frames:
        monkeys = set()
        for cd in det2d.values():
            if frame in cd:
                monkeys.update(cd[frame].keys())
        frame_out = {}
        for monkey in monkeys:
            kps_3d, any_valid = [], False
            bbox_diags = []
            for ki in range(17):
                cam_Ps, cam_pts = [], []
                for cam_id, cd in det2d.items():
                    if cam_id not in Ps or frame not in cd or monkey not in cd[frame]:
                        continue
                    kp = cd[frame][monkey]["kps"][ki]
                    if kp[2] >= KP_CONF_THRESH:
                        cam_Ps.append(Ps[cam_id])
                        cam_pts.append(kp[:2])
                if len(cam_Ps) >= MIN_CAMERAS:
                    kps_3d.append([round(v, 1) for v in _triangulate(cam_Ps, cam_pts)])
                    any_valid = True
                else:
                    kps_3d.append(None)
            for cam_id, cd in det2d.items():
                if frame in cd and monkey in cd[frame]:
                    b = cd[frame][monkey]["bbox"]
                    bbox_diags.append(np.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2))
            if any_valid:
                bbox_Ps, bbox_pts = [], []
                for cam_id, cd in det2d.items():
                    if cam_id not in Ps or frame not in cd or monkey not in cd[frame]:
                        continue
                    b = cd[frame][monkey]["bbox"]
                    bbox_Ps.append(Ps[cam_id])
                    bbox_pts.append([(b[0]+b[2])/2.0, (b[1]+b[3])/2.0])
                if len(bbox_Ps) >= MIN_CAMERAS:
                    com = [round(v, 1) for v in _triangulate(bbox_Ps, bbox_pts)]
                else:
                    hips = [kps_3d[i] for i in [11, 12] if kps_3d[i] is not None]
                    com  = ([round(sum(h[j] for h in hips)/len(hips), 1)
                             for j in range(3)] if hips else None)
                avg_diag = float(np.mean(bbox_diags)) if bbox_diags else 400.0
                frame_out[monkey] = {"kps": kps_3d, "com": com, "bbox_diag": avg_diag}
        if frame_out:
            frames_out[frame] = frame_out
    return frames_out


# ─── PyQt6 + pyqtgraph GUI ────────────────────────────────────────────────────
def run_viewer(frames_out, det2d, videos, calib, session_name, folder):
    import cv2
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QGridLayout, QLabel, QPushButton, QSlider, QCheckBox,
        QGroupBox, QSizePolicy, QComboBox,
    )
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    frame_nums  = sorted(frames_out.keys())
    monkey_ids  = sorted({m for fd in frames_out.values() for m in fd})
    n_frames    = len(frame_nums)
    rx, ry, rz  = ROOM["x"], ROOM["y"], ROOM["z"]

    # ── Camera positions in both frames ──────────────────────────────────────
    cam_pos_recon    = {}
    cam_pos_physical = {}
    cam_R            = {}
    for cid, info in calib["cameras"].items():
        if "reconstructed_aligned_mm" in info:
            cam_pos_recon[cid]    = list(info["reconstructed_aligned_mm"])
        if "known_mm" in info:
            cam_pos_physical[cid] = list(info["known_mm"])
        if "R_world" in info:
            cam_R[cid] = np.array(info["R_world"])

    _src = [cam_pos_recon[c]    for c in CAM_ORDER
            if c in cam_pos_recon and c in cam_pos_physical]
    _dst = [cam_pos_physical[c] for c in CAM_ORDER
            if c in cam_pos_recon and c in cam_pos_physical]
    procrustes_params = compute_procrustes(_src, _dst) if len(_src) >= 3 else None

    frame_mode = [1]   # 0 = Reconstructed, 1 = Physical (default)

    # ── Session statistics ────────────────────────────────────────────────────
    good_frame_indices = set()
    path_mm  = {mid: 0.0 for mid in monkey_ids}
    prev_com = {mid: None for mid in monkey_ids}
    min_prox = float('inf')

    for fi, fn in enumerate(frame_nums):
        fd = frames_out.get(fn, {})
        coms = {}
        for mid in monkey_ids:
            md = fd.get(mid)
            if md and md.get('com'):
                coms[mid] = np.array(md['com'], dtype=float)
        if len(monkey_ids) >= 2 and len(coms) == len(monkey_ids):
            good_frame_indices.add(fi)
        elif len(monkey_ids) == 1 and coms:
            good_frame_indices.add(fi)
        for mid, com in coms.items():
            if prev_com[mid] is not None:
                path_mm[mid] += float(np.linalg.norm(com - prev_com[mid]))
            prev_com[mid] = com
        if len(monkey_ids) >= 2 and len(coms) == 2:
            mids = list(coms.keys())
            d = float(np.linalg.norm(coms[mids[0]] - coms[mids[1]]))
            if d < min_prox:
                min_prox = d

    path_strs = ", ".join(
        f"<b style='color:{MONKEY_COLORS_QT[i]}'>{mid}</b> {path_mm[mid]/1000:.1f}m"
        for i, mid in enumerate(monkey_ids))
    prox_str  = f"{min_prox/10:.0f} cm" if min_prox < float('inf') else "—"
    stats_html = (
        f"Path: {path_strs}  ·  Min proximity: {prox_str}  ·  "
        f"Reconstructed: {len(good_frame_indices)}/{n_frames} frames"
    )

    # ── Top-5 uninterrupted runs (both monkeys tracked) ───────────────────────
    good_sorted = sorted(good_frame_indices)
    top5_runs = []   # list of (run_len, start_idx, end_idx)
    if good_sorted:
        rs, rl = good_sorted[0], 1
        for i in range(1, len(good_sorted)):
            # consecutive if frame numbers are adjacent
            if frame_nums[good_sorted[i]] == frame_nums[good_sorted[i-1]] + 1:
                rl += 1
            else:
                top5_runs.append((rl, rs, good_sorted[i-1]))
                rs, rl = good_sorted[i], 1
        top5_runs.append((rl, rs, good_sorted[-1]))
        top5_runs.sort(reverse=True)
        top5_runs = top5_runs[:5]

    # ── Top-5 runs: single animal seen by ≥3 cameras ─────────────────────────
    # Count cameras detecting each monkey per frame
    cam_count = {}   # frame_num → {monkey_id: n_cameras}
    for cam_id, cam_data in det2d.items():
        for fn, fd in cam_data.items():
            if fn not in cam_count:
                cam_count[fn] = {}
            for mid in fd:
                cam_count[fn][mid] = cam_count[fn].get(mid, 0) + 1

    single_good_indices = set()
    for fi, fn in enumerate(frame_nums):
        counts = cam_count.get(fn, {})
        if any(c >= 3 for c in counts.values()):
            single_good_indices.add(fi)

    single_sorted = sorted(single_good_indices)
    top5_single_runs = []
    if single_sorted:
        rs, rl = single_sorted[0], 1
        for i in range(1, len(single_sorted)):
            if frame_nums[single_sorted[i]] == frame_nums[single_sorted[i-1]] + 1:
                rl += 1
            else:
                top5_single_runs.append((rl, rs, single_sorted[i-1]))
                rs, rl = single_sorted[i], 1
        top5_single_runs.append((rl, rs, single_sorted[-1]))
        top5_single_runs.sort(reverse=True)
        top5_single_runs = top5_single_runs[:5]

    # ── Camera cone mesh builder ──────────────────────────────────────────────
    def make_cone_meshdata(pos, R, radius=60, height=120, n=20):
        """World-space cone: base at pos, tip along optical axis R[2,:]."""
        pos = np.array(pos, dtype=float)
        fwd = np.array(R)[2]                          # optical axis in world
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        # Build local frame perpendicular to fwd
        ref  = np.array([0, 0, 1.0]) if abs(fwd[2]) < 0.9 else np.array([1, 0, 0.0])
        perp1 = np.cross(fwd, ref);  perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(fwd, perp1)
        # Base circle vertices
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        base = np.array([pos + radius*(np.cos(a)*perp1 + np.sin(a)*perp2)
                         for a in angles])
        tip    = pos + fwd * height
        center = pos
        # vertex layout: [base[0..n-1], tip, center]  (n+2 vertices)
        verts = np.vstack([base, [tip], [center]])
        faces = []
        for i in range(n):
            j = (i + 1) % n
            faces.append([i, j, n])      # side: base[i], base[j], tip
            faces.append([n+1, j, i])    # end cap: center, base[j], base[i]
        return gl.MeshData(vertexes=verts, faces=np.array(faces))

    # ── cv2 captures ─────────────────────────────────────────────────────────
    caps = {}
    for cid in CAM_ORDER:
        if cid in videos:
            cap = cv2.VideoCapture(str(videos[cid]))
            if cap.isOpened():
                caps[cid] = cap

    def read_frame(cid, frame_num):
        cap = caps.get(cid)
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, img = cap.read()
        return img if ok else None

    # ── Window ───────────────────────────────────────────────────────────────
    win = QMainWindow()
    win.setWindowTitle(f"HomeCage 3D · {session_name}  ({folder.name})")
    win.resize(1700, 950)

    central = QWidget()
    win.setCentralWidget(central)
    root_v = QVBoxLayout(central)
    root_v.setContentsMargins(4, 4, 4, 4)
    root_v.setSpacing(2)

    # top row: 3D view (left) + controls/videos (right)
    top_row = QWidget()
    root = QHBoxLayout(top_row)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(4)
    root_v.addWidget(top_row, stretch=1)

    # ── Left: GL view + stats ─────────────────────────────────────────────────
    left = QWidget()
    left_v = QVBoxLayout(left)
    left_v.setContentsMargins(0, 0, 0, 0)
    left_v.setSpacing(2)

    gl_view = gl.GLViewWidget()
    gl_view.setBackgroundColor((12, 12, 12, 255))
    left_v.addWidget(gl_view)

    # info overlay (absolute inside gl_view)
    info_lbl = QLabel(gl_view)
    info_lbl.setStyleSheet(
        "background:rgba(0,0,0,160);color:#aac8ff;"
        "font:10px monospace;padding:6px 10px;border-radius:6px;")
    info_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
    info_lbl.move(8, 8)
    info_lbl.setText(
        f"<b>Session:</b> {session_name}<br>"
        f"<b>Folder:</b> {folder.name}<br>"
        f"<b>Room:</b> {rx}×{ry}×{rz} mm<br>"
        f"<b>Cameras:</b> {', '.join(CAM_ORDER)}<br>"
        f"<b>Monkeys:</b> {', '.join(monkey_ids)}<br>"
        f"<b>Frames:</b> {n_frames}"
    )
    info_lbl.adjustSize()

    # stats bar (below gl_view, above ctrl_bar)
    stats_lbl = QLabel()
    stats_lbl.setFixedHeight(20)
    stats_lbl.setStyleSheet(
        "background:#0d1a0d;color:#88cc88;font:9px monospace;padding:2px 6px;")
    stats_lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
    stats_lbl.setText(stats_html)
    left_v.addWidget(stats_lbl)

    # ── Full-width playback bar (added to root_v after top_row) ───────────────
    ctrl_bar = QWidget()
    ctrl_bar.setFixedHeight(38)
    ctrl_h = QHBoxLayout(ctrl_bar)
    ctrl_h.setContentsMargins(4, 0, 4, 0)
    ctrl_h.setSpacing(6)

    btn_play = QPushButton("▶")
    btn_play.setFixedSize(28, 34)
    btn_play.setStyleSheet("font-size:11px;padding:0;")

    # Slider container: ticks bar (8px) stacked above slider (18px)
    class FrameTicksBar(QWidget):
        def __init__(self):
            super().__init__()
            self.setFixedHeight(8)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Click to jump to nearest reconstructed frame")

        def paintEvent(self, _e):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(18, 18, 18))
            if n_frames <= 1:
                return
            w = max(1, self.width() - 1)
            for idx in good_frame_indices:
                x = int(idx / (n_frames - 1) * w)
                p.fillRect(x, 0, 2, 8, QColor(60, 210, 60, 210))

        def mousePressEvent(self, e):
            if n_frames <= 1:
                return
            raw = round(e.pos().x() / max(1, self.width()-1) * (n_frames-1))
            raw = max(0, min(n_frames-1, raw))
            target = (min(good_frame_indices, key=lambda i: abs(i-raw))
                      if good_frame_indices else raw)
            set_frame(target)

    slider_container = QWidget()
    slider_v = QVBoxLayout(slider_container)
    slider_v.setContentsMargins(0, 0, 0, 0)
    slider_v.setSpacing(1)

    ticks_bar = FrameTicksBar()
    slider_v.addWidget(ticks_bar)

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, n_frames - 1)
    slider.setFixedHeight(18)
    slider_v.addWidget(slider)

    lbl_frame = QLabel("#—")
    lbl_frame.setFixedWidth(120)
    lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl_frame.setStyleSheet("font-size:10px;")
    speed_combo = QComboBox()
    speed_combo.setFixedSize(60, 28)
    speed_combo.setStyleSheet("font-size:10px;")
    for lbl_s, val in [("0.25×", .25), ("0.5×", .5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0)]:
        speed_combo.addItem(lbl_s, val)
    speed_combo.setCurrentIndex(2)

    btn_rec = QPushButton("⏺ Rec")
    btn_rec.setFixedSize(56, 28)
    btn_rec.setStyleSheet("font-size:10px;padding:0;color:#ff4444;")

    ctrl_h.addWidget(btn_play)
    ctrl_h.addWidget(slider_container, stretch=1)
    ctrl_h.addWidget(lbl_frame)
    ctrl_h.addWidget(speed_combo)
    ctrl_h.addWidget(btn_rec)

    root.addWidget(left, stretch=3)

    # ── Right: toggles + video grid ───────────────────────────────────────────
    right = QWidget()
    right.setFixedWidth(660)
    right_v = QVBoxLayout(right)
    right_v.setContentsMargins(0, 0, 0, 0)
    right_v.setSpacing(4)

    tog_box = QGroupBox("Display")
    tog_v = QVBoxLayout(tog_box)
    tog_v.setSpacing(2)

    def make_cb(label, checked=True):
        cb = QCheckBox(label); cb.setChecked(checked); return cb

    cb_skel   = make_cb("Skeleton bones")
    cb_dots   = make_cb("Keypoints")
    cb_com    = make_cb("CoM sphere")
    cb_trail  = make_cb("Trail", False)
    cb_videos = make_cb("Video feeds")

    trail_combo = QComboBox()
    for lbl_t, val in [("30 f", 30), ("60 f", 60), ("2 min", 120), ("5 min", 300), ("∞", 99999)]:
        trail_combo.addItem(lbl_t, val)
    trail_combo.setCurrentIndex(1)

    trail_row = QWidget()
    tr_h = QHBoxLayout(trail_row)
    tr_h.setContentsMargins(0, 0, 0, 0)
    tr_h.addWidget(cb_trail); tr_h.addWidget(trail_combo); tr_h.addStretch()

    monkey_cbs = []
    for mi, mid in enumerate(monkey_ids):
        color = MONKEY_COLORS_QT[mi % len(MONKEY_COLORS_QT)]
        cb = make_cb(f"● {mid}")
        cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
        monkey_cbs.append(cb)

    mode_combo = QComboBox()
    mode_combo.addItem("Mode A — Reconstructed  (internally consistent)", 0)
    mode_combo.addItem("Mode B — Physical  (Procrustes-corrected)", 1)
    mode_combo.setCurrentIndex(1)
    mode_combo.setStyleSheet("font-size:10px;")

    for w in [mode_combo, cb_skel, cb_dots, cb_com, trail_row]:
        tog_v.addWidget(w)
    tog_v.addWidget(QLabel("── Monkeys ──"))
    for cb in monkey_cbs:
        tog_v.addWidget(cb)
    tog_v.addWidget(QLabel("── Cameras  (click video → snap 3D view) ──"))
    tog_v.addWidget(cb_videos)

    # ── Top-5 sequences panel (own QGroupBox, right of Display) ──────────────
    seq_box = QGroupBox("Best sequences")
    seq_v   = QVBoxLayout(seq_box)
    seq_v.setSpacing(6)
    seq_v.setContentsMargins(4, 6, 4, 6)

    def _seq_label(fn_start, rl):
        secs      = rl / 25.0
        t_m, t_s  = divmod(int(fn_start / 25.0), 60)
        return (f"frame {fn_start}  ({t_m:02d}:{t_s:02d})\n"
                f"{rl} frames · {secs:.1f}s")

    lbl_both = QLabel("● Both animals, ≥2 cams")
    lbl_both.setStyleSheet("font-size:9px; color:#88ccff; padding:2px 0;")
    seq_v.addWidget(lbl_both)

    if top5_runs:
        for rank, (rl, start_idx, _end) in enumerate(top5_runs, 1):
            btn = QPushButton(f"#{rank}  {_seq_label(frame_nums[start_idx], rl)}")
            btn.setStyleSheet(
                "font-size:10px; padding:6px 8px;"
                "background:#0d1f33; color:#88ccff;")
            btn.clicked.connect(lambda _c, idx=start_idx: set_frame(idx))
            seq_v.addWidget(btn)
    else:
        seq_v.addWidget(QLabel("No runs found"))

    lbl_single = QLabel("● Single animal, ≥3 cams")
    lbl_single.setStyleSheet("font-size:9px; color:#ffcc44; padding:2px 0;")
    seq_v.addWidget(lbl_single)

    if top5_single_runs:
        for rank, (rl, start_idx, _end) in enumerate(top5_single_runs, 1):
            btn = QPushButton(f"#{rank}  {_seq_label(frame_nums[start_idx], rl)}")
            btn.setStyleSheet(
                "font-size:10px; padding:6px 8px;"
                "background:#2a1e00; color:#ffcc44;")
            btn.clicked.connect(lambda _c, idx=start_idx: set_frame(idx))
            seq_v.addWidget(btn)
    else:
        seq_v.addWidget(QLabel("No runs found"))

    seq_v.addStretch()

    # top bar of the right panel: Display | Best sequences side by side
    top_controls = QWidget()
    top_ctrl_h   = QHBoxLayout(top_controls)
    top_ctrl_h.setContentsMargins(0, 0, 0, 0)
    top_ctrl_h.setSpacing(4)
    top_ctrl_h.addWidget(tog_box, stretch=3)
    top_ctrl_h.addWidget(seq_box, stretch=2)
    right_v.addWidget(top_controls)

    # ── 2×2 video grid ───────────────────────────────────────────────────────
    class ClickLabel(QLabel):
        def __init__(self, cid, *a, **kw):
            super().__init__(*a, **kw)
            self.cid = cid
        def mousePressEvent(self, _):
            snap_camera(self.cid)

    vid_grid_widget = QWidget()
    vid_grid = QGridLayout(vid_grid_widget)
    vid_grid.setContentsMargins(0, 0, 0, 0)
    vid_grid.setSpacing(2)

    vid_labels = {}
    for i, cid in enumerate(CAM_ORDER):
        row, col = divmod(i, 2)
        lbl = ClickLabel(cid)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"background:#111; border:2px solid {CAM_COLORS_QT[cid]};")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumSize(200, 150)
        lbl.setText(f"Cam {cid}")
        vid_grid.addWidget(lbl, row, col)
        vid_labels[cid] = lbl

    right_v.addWidget(vid_grid_widget, stretch=1)
    root.addWidget(right)

    # ctrl_bar spans full window width at the bottom
    root_v.addWidget(ctrl_bar)

    # ── 3D scene ─────────────────────────────────────────────────────────────
    # Room wireframe
    corners = np.array([[x, y, z]
                        for x in [0, rx] for y in [0, ry] for z in [0, rz]])
    for a, b in [(0,1),(0,2),(0,4),(1,3),(1,5),(2,3),(2,6),(3,7),
                 (4,5),(4,6),(5,7),(6,7)]:
        gl_view.addItem(gl.GLLinePlotItem(
            pos=np.array([corners[a], corners[b]], dtype=float),
            color=(0.13, 0.27, 0.4, 0.5), width=1, antialias=True))

    grid = gl.GLGridItem()
    grid.setSize(x=rx, y=ry)
    grid.setSpacing(x=rx/10, y=ry/10)
    grid.translate(rx/2, ry/2, 0)
    gl_view.addItem(grid)

    # ── Camera cones (red, fixed in world space) ──────────────────────────────
    cam_cone_items = {}   # GLMeshItem per camera
    cam_text_items = {}   # GLTextItem per camera

    CONE_COLOR = (1.0, 0.18, 0.18, 0.88)

    def _cam_pos(cid, mode):
        if mode == 0:
            return cam_pos_recon.get(cid, list(CAM_KNOWN_POS[cid]))
        return cam_pos_physical.get(cid, list(CAM_KNOWN_POS[cid]))

    for cid in CAM_ORDER:
        R   = cam_R.get(cid, np.eye(3))
        pos = _cam_pos(cid, frame_mode[0])
        md  = make_cone_meshdata(pos, R)
        cone = gl.GLMeshItem(meshdata=md, color=CONE_COLOR,
                             smooth=True, drawEdges=False,
                             glOptions='translucent')
        gl_view.addItem(cone)
        cam_cone_items[cid] = cone

        try:
            px, py, pz = pos
            txt = gl.GLTextItem(
                pos=np.array([px, py, pz + 160], dtype=float),
                text=f"Cam {cid}",
                color=(255, 80, 80, 255))
            gl_view.addItem(txt)
            cam_text_items[cid] = txt
        except Exception:
            pass

    def update_cam_markers():
        for cid in CAM_ORDER:
            R   = cam_R.get(cid, np.eye(3))
            pos = _cam_pos(cid, frame_mode[0])
            md  = make_cone_meshdata(pos, R)
            cam_cone_items[cid].setMeshData(meshdata=md)
            if cid in cam_text_items:
                px, py, pz = pos
                cam_text_items[cid].setData(
                    pos=np.array([px, py, pz + 160], dtype=float))

    # ── Per-monkey 3D objects ─────────────────────────────────────────────────
    class MonkeyObjs:
        def __init__(self, mi, mid):
            self.mid   = mid
            self.col   = MONKEY_COLORS_F[mi % len(MONKEY_COLORS_F)]
            self.trail_pts = []
            self.visible   = True
            _o = np.zeros((1, 3), dtype=float)

            self.bones = []
            for _ in SKEL:
                ln = gl.GLLinePlotItem(
                    pos=np.zeros((2, 3), dtype=float),
                    color=self.col, width=2, antialias=True)
                ln.setVisible(False)
                gl_view.addItem(ln)
                self.bones.append(ln)

            self.dots = gl.GLScatterPlotItem(
                pos=_o, color=np.array([self.col]), size=8, pxMode=True)
            self.dots.setVisible(False)
            gl_view.addItem(self.dots)

            self.com_item = gl.GLScatterPlotItem(
                pos=_o, color=np.array([self.col]), size=250, pxMode=False)
            self.com_item.setVisible(False)
            gl_view.addItem(self.com_item)

            self.trail_line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3), dtype=float),
                color=(*self.col[:3], 0.45), width=2, antialias=True)
            self.trail_line.setVisible(False)
            gl_view.addItem(self.trail_line)

        def update(self, kps, com, bbox_diag):
            if not self.visible:
                for b in self.bones: b.setVisible(False)
                self.dots.setVisible(False)
                self.com_item.setVisible(False)
                self.trail_line.setVisible(False)
                return

            if com:
                self.trail_pts.append(com[:])
                tl = trail_combo.currentData()
                if len(self.trail_pts) > tl:
                    self.trail_pts = self.trail_pts[-tl:]

            if len(self.trail_pts) > 1 and cb_trail.isChecked():
                self.trail_line.setData(pos=np.array(self.trail_pts, dtype=float))
                self.trail_line.setVisible(True)
            else:
                self.trail_line.setVisible(False)

            if com and cb_com.isChecked():
                com_size = max(200.0, min(600.0, bbox_diag * 0.55))
                self.com_item.setData(pos=np.array([com], dtype=float), size=com_size)
                self.com_item.setVisible(True)
            else:
                self.com_item.setVisible(False)

            if kps:
                valid_pts = [kp for kp in kps if kp is not None]
                if valid_pts and cb_dots.isChecked():
                    self.dots.setData(pos=np.array(valid_pts, dtype=float),
                                      color=np.array([self.col]*len(valid_pts)))
                    self.dots.setVisible(True)
                else:
                    self.dots.setVisible(False)
                for bi, (a, b) in enumerate(SKEL):
                    pa, pb = kps[a], kps[b]
                    if pa is not None and pb is not None and cb_skel.isChecked():
                        self.bones[bi].setData(pos=np.array([pa, pb], dtype=float))
                        self.bones[bi].setVisible(True)
                    else:
                        self.bones[bi].setVisible(False)
            else:
                self.dots.setVisible(False)
                for b in self.bones:
                    b.setVisible(False)

    monkey_objs = [MonkeyObjs(mi, mid) for mi, mid in enumerate(monkey_ids)]

    # ── Camera snap ──────────────────────────────────────────────────────────
    room_center = np.array([rx/2, ry/2, rz/4])

    def snap_camera(cid):
        if frame_mode[0] == 1:
            pos = cam_pos_physical.get(cid)
            if pos is None:
                return
            C = np.array(pos, dtype=float)
        else:
            C, _ = cam_world_pose(calib, cid)
            if C is None:
                return
        from_center = C - room_center
        dist = float(np.linalg.norm(from_center))
        if dist < 1:
            return
        d       = from_center / dist
        elev    = float(np.degrees(np.arcsin(np.clip(d[2], -1, 1))))
        azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
        gl_view.opts['center'] = pg.Vector(*room_center)
        gl_view.setCameraPosition(distance=dist, elevation=elev, azimuth=azimuth)
        gl_view.update()

    # ── Frame update ──────────────────────────────────────────────────────────
    cur_idx = [0]

    def _xform_pt(pt):
        if frame_mode[0] == 0 or procrustes_params is None:
            return pt
        return apply_procrustes(pt, *procrustes_params)

    def _xform_kps(kps):
        return [_xform_pt(kp) for kp in kps] if kps else None

    def set_frame(idx):
        cur_idx[0] = idx
        slider.blockSignals(True)
        slider.setValue(idx)
        slider.blockSignals(False)
        frame_num = frame_nums[idx]
        lbl_frame.setText(f"#{frame_num}  {idx+1}/{n_frames}")

        fd = frames_out.get(frame_num, {})
        parts = []
        for mi, (obj, mid) in enumerate(zip(monkey_objs, monkey_ids)):
            obj.visible = monkey_cbs[mi].isChecked()
            md = fd.get(mid)
            com_raw = _xform_pt(md["com"]) if md else None
            obj.update(
                _xform_kps(md["kps"]) if md else None,
                com_raw,
                md["bbox_diag"] if md else 300.0,
            )
            if com_raw:
                c = com_raw
                parts.append(f"<b style='color:{MONKEY_COLORS_QT[mi]}'>"
                              f"{mid}</b> ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})")

        mode_str = "A · Reconstructed" if frame_mode[0] == 0 else "B · Physical"
        info_lbl.setText(
            f"<b>Session:</b> {session_name}<br>"
            f"<b>Folder:</b> {folder.name}<br>"
            f"<b>Mode:</b> {mode_str}<br>"
            f"<b>Room:</b> {rx}×{ry}×{rz} mm<br>"
            f"<b>Cameras:</b> {', '.join(CAM_ORDER)}<br>"
            f"<b>Monkeys:</b> {', '.join(monkey_ids)}<br>"
            f"<b>Frame:</b> #{frame_num}  ({idx+1}/{n_frames})<br>"
            + ("<br>".join(parts) if parts else "<i>no detection</i>")
        )
        info_lbl.adjustSize()

        if cb_videos.isChecked():
            update_videos(frame_num)

    def update_videos(frame_num):
        for cid in CAM_ORDER:
            lbl = vid_labels[cid]
            w, h = lbl.width(), lbl.height()
            if w < 10 or h < 10:
                continue
            img = read_frame(cid, frame_num)
            if img is None:
                lbl.setText(f"Cam {cid}\n(no frame)")
                continue
            cv2.putText(img, f"Cam {cid}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 6)
            col_bgr = tuple(int(c*255) for c in reversed(CAM_COLORS_F[cid][:3]))
            cv2.putText(img, f"Cam {cid}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col_bgr, 3)
            scale = min(w / VIDEO_W, h / VIDEO_H)
            nw, nh = max(1, int(VIDEO_W*scale)), max(1, int(VIDEO_H*scale))
            img_rgb = cv2.cvtColor(cv2.resize(img, (nw, nh)), cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, nw, nh,
                          img_rgb.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap(w, h)
            pix.fill(QColor(17, 17, 17))
            p = QPainter(pix)
            p.drawPixmap((w-nw)//2, (h-nh)//2, QPixmap.fromImage(qimg))
            p.end()
            lbl.setPixmap(pix)

    # ── Recording ─────────────────────────────────────────────────────────────
    recording    = [False]
    video_writer = [None]

    def start_recording():
        ts     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_p  = str(folder / f"recording_{session_name}_{ts}.mp4")
        ww, wh = win.width(), win.height()
        fps    = max(1.0, speed_combo.currentData() * 5.0)
        vw     = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*'mp4v'), fps, (ww, wh))
        if not vw.isOpened():
            print(f"ERROR: could not open video writer"); return
        video_writer[0] = vw
        recording[0]    = True
        btn_rec.setText("⏹ Stop")
        btn_rec.setStyleSheet("font-size:10px;padding:0;color:#ff0000;background:#220000;")
        print(f"Recording → {out_p}")

    def stop_recording():
        recording[0] = False
        if video_writer[0] is not None:
            video_writer[0].release(); video_writer[0] = None
        btn_rec.setText("⏺ Rec")
        btn_rec.setStyleSheet("font-size:10px;padding:0;color:#ff4444;")
        print("Recording stopped.")

    def capture_window():
        if not recording[0] or video_writer[0] is None:
            return
        pix = win.grab()
        img = pix.toImage().convertToFormat(QImage.Format.Format_RGB888)
        w, h = img.width(), img.height()
        ptr = img.bits(); ptr.setsize(h * w * 3)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
        video_writer[0].write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    btn_rec.clicked.connect(lambda: stop_recording() if recording[0] else start_recording())

    # ── Playback ─────────────────────────────────────────────────────────────
    playing = [False]
    timer   = QTimer()

    def tick():
        speed = speed_combo.currentData()
        tick.skip = getattr(tick, 'skip', 0)
        if speed < 1:
            tick.skip += 1
            if tick.skip < int(round(1/speed)):
                return
            tick.skip = 0
        set_frame((cur_idx[0] + max(1, int(round(speed)))) % n_frames)
        capture_window()

    timer.timeout.connect(tick)

    def toggle_play():
        playing[0] = not playing[0]
        if playing[0]:
            timer.setInterval(max(40, int(200 / speed_combo.currentData())))
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
                   update_videos(frame_nums[cur_idx[0]])
                   if cb_videos.isChecked() else None))

    def on_mode_change(_):
        frame_mode[0] = mode_combo.currentData()
        for obj in monkey_objs:
            obj.trail_pts = []
        update_cam_markers()
        set_frame(cur_idx[0])

    mode_combo.currentIndexChanged.connect(on_mode_change)

    for cb in monkey_cbs + [cb_skel, cb_dots, cb_com, cb_trail]:
        cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))
    trail_combo.currentIndexChanged.connect(lambda _: set_frame(cur_idx[0]))

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    _key_cam_map = {
        Qt.Key.Key_1: CAM_ORDER[0],
        Qt.Key.Key_2: CAM_ORDER[1],
        Qt.Key.Key_3: CAM_ORDER[2],
        Qt.Key.Key_4: CAM_ORDER[3],
    }

    def _key_press(event):
        key = event.key()
        if key == Qt.Key.Key_Space:
            toggle_play()
        elif key in _key_cam_map:
            snap_camera(_key_cam_map[key])
        else:
            type(win).keyPressEvent(win, event)

    win.keyPressEvent = _key_press

    # ── Show ─────────────────────────────────────────────────────────────────
    win.show()
    gl_view.opts['center'] = pg.Vector(rx/2, ry/2, rz/4)
    gl_view.setCameraPosition(distance=max(rx, ry, rz)*3.0, elevation=30, azimuth=45)
    set_frame(0)
    sys.exit(app.exec())


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)

    folder = Path(sys.argv[1])
    if not folder.exists():
        print(f"ERROR: {folder} does not exist"); sys.exit(1)
    if not CALIB_F.exists():
        print(f"ERROR: calibration not found at {CALIB_F}"); sys.exit(1)

    session_name = folder.parent.name
    print(f"Session: {session_name}  ({folder})")

    print("\n── Calibration ──")
    calib = load_calib()
    Ps    = build_projection_matrices(calib)
    print(f"  Cameras: {sorted(Ps.keys())}")

    print("\n── Videos ──")
    videos = find_videos(folder)
    for cid, vpath in sorted(videos.items()):
        print(f"  cam{cid}: {vpath.name}")

    print("\n── 2D results ──")
    det2d = parse_2d_results(folder)

    print("\n── Triangulating ──")
    frames_out = triangulate_session(det2d, Ps)
    print(f"  Frames with 3D output: {len(frames_out)}")

    print("\n── Launching viewer ──")
    run_viewer(frames_out, det2d, videos, calib, session_name, folder)
