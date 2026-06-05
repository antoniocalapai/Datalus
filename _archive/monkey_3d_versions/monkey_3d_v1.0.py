#!/usr/bin/env python3
"""
HomeCage — Monkey 3D Triangulation & Viewer  (PyQt6 + pyqtgraph + cv2)

Usage:
    python3 monkey_3d.py <2d_results_folder>
"""

import sys, json, re
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
    """Return (C_world, R_world) – camera centre and rotation matrix."""
    info = calib["cameras"].get(cid, {})
    if "R_world" not in info:
        return None, None
    R = np.array(info["R_world"])
    T = np.array(info["T_world"])
    C = -R.T @ T
    return C, R


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
            # average bbox diagonal across cameras for CoM sphere sizing
            for cam_id, cd in det2d.items():
                if frame in cd and monkey in cd[frame]:
                    b = cd[frame][monkey]["bbox"]
                    bbox_diags.append(np.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2))
            if any_valid:
                # triangulate CoM from bbox centres across cameras
                bbox_Ps, bbox_pts = [], []
                for cam_id, cd in det2d.items():
                    if cam_id not in Ps or frame not in cd or monkey not in cd[frame]:
                        continue
                    b = cd[frame][monkey]["bbox"]
                    cx2d = (b[0] + b[2]) / 2.0
                    cy2d = (b[1] + b[3]) / 2.0
                    bbox_Ps.append(Ps[cam_id])
                    bbox_pts.append([cx2d, cy2d])
                if len(bbox_Ps) >= MIN_CAMERAS:
                    com = [round(v, 1) for v in _triangulate(bbox_Ps, bbox_pts)]
                else:
                    # fallback: average hip keypoints
                    hips = [kps_3d[i] for i in [11, 12] if kps_3d[i] is not None]
                    com  = [round(sum(h[j] for h in hips)/len(hips), 1)
                            for j in range(3)] if hips else None
                avg_diag = float(np.mean(bbox_diags)) if bbox_diags else 400.0
                frame_out[monkey] = {"kps": kps_3d, "com": com,
                                     "bbox_diag": avg_diag}
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
    from PyQt6.QtGui import QImage, QPixmap, QColor, QPainter, QFont, QPen
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    frame_nums  = sorted(frames_out.keys())
    monkey_ids  = sorted({m for fd in frames_out.values() for m in fd})
    n_frames    = len(frame_nums)
    rx, ry, rz = ROOM["x"], ROOM["y"], ROOM["z"]

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
    root = QHBoxLayout(central)
    root.setContentsMargins(4, 4, 4, 4)
    root.setSpacing(4)

    # ── Left: GL view + controls ──────────────────────────────────────────────
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

    # compact playback bar
    ctrl_bar = QWidget()
    ctrl_bar.setFixedHeight(28)
    ctrl_h = QHBoxLayout(ctrl_bar)
    ctrl_h.setContentsMargins(4, 0, 4, 0)
    ctrl_h.setSpacing(6)

    btn_play = QPushButton("▶")
    btn_play.setFixedSize(28, 22)
    btn_play.setStyleSheet("font-size:11px;padding:0;")
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, n_frames - 1)
    slider.setFixedHeight(18)
    lbl_frame = QLabel("#—")
    lbl_frame.setFixedWidth(120)
    lbl_frame.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl_frame.setStyleSheet("font-size:10px;")
    speed_combo = QComboBox()
    speed_combo.setFixedSize(60, 22)
    speed_combo.setStyleSheet("font-size:10px;")
    for lbl_s, val in [("0.25×", .25), ("0.5×", .5), ("1×", 1.0), ("2×", 2.0), ("5×", 5.0)]:
        speed_combo.addItem(lbl_s, val)
    speed_combo.setCurrentIndex(2)

    ctrl_h.addWidget(btn_play)
    ctrl_h.addWidget(slider, stretch=1)
    ctrl_h.addWidget(lbl_frame)
    ctrl_h.addWidget(speed_combo)
    left_v.addWidget(ctrl_bar)

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
    cb_trail  = make_cb("Trail")
    cb_videos = make_cb("Video feeds")
    cb_2d     = make_cb("2D overlay", False)  # off by default; video already has ABT annotations

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

    for w in [cb_skel, cb_dots, cb_com, trail_row]:
        tog_v.addWidget(w)
    tog_v.addWidget(QLabel("── Monkeys ──"))
    for cb in monkey_cbs:
        tog_v.addWidget(cb)
    tog_v.addWidget(QLabel("── Cameras  (click video → snap 3D view) ──"))
    tog_v.addWidget(cb_videos)
    tog_v.addWidget(cb_2d)
    right_v.addWidget(tog_box)

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
        lbl.setStyleSheet(
            f"background:#111; border:2px solid {CAM_COLORS_QT[cid]};")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumSize(200, 150)
        lbl.setText(f"Cam {cid}")
        vid_grid.addWidget(lbl, row, col)
        vid_labels[cid] = lbl

    right_v.addWidget(vid_grid_widget, stretch=1)
    root.addWidget(right)

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

    # Camera markers + labels
    for cid in CAM_ORDER:
        px, py, pz = CAM_KNOWN_POS[cid]
        col = CAM_COLORS_F[cid]
        sp = gl.GLScatterPlotItem(
            pos=np.array([[px, py, pz]], dtype=float),
            color=np.array([col]), size=18, pxMode=True)
        gl_view.addItem(sp)
        # vertical drop line
        gl_view.addItem(gl.GLLinePlotItem(
            pos=np.array([[px, py, 0], [px, py, pz]], dtype=float),
            color=(*col[:3], 0.3), width=1, antialias=True))
        try:
            txt = gl.GLTextItem(
                pos=np.array([px, py, pz + 120], dtype=float),
                text=f"Cam {cid}",
                color=(*[int(c*255) for c in col[:3]], 255))
            gl_view.addItem(txt)
        except Exception:
            pass  # GLTextItem not available in all pyqtgraph builds

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

            # CoM: world-space sphere so size reflects body scale
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

            # trail
            if com:
                self.trail_pts.append(com[:])
                tl = trail_combo.currentData()
                if len(self.trail_pts) > tl:
                    self.trail_pts = self.trail_pts[-tl:]

            if len(self.trail_pts) > 1 and cb_trail.isChecked():
                self.trail_line.setData(
                    pos=np.array(self.trail_pts, dtype=float))
                self.trail_line.setVisible(True)
            else:
                self.trail_line.setVisible(False)

            if com and cb_com.isChecked():
                # scale CoM sphere by bbox diagonal (world-space mm)
                com_size = max(200.0, min(600.0, bbox_diag * 0.55))
                self.com_item.setData(
                    pos=np.array([com], dtype=float),
                    size=com_size)
                self.com_item.setVisible(True)
            else:
                self.com_item.setVisible(False)

            if kps:
                valid_pts = [kp for kp in kps if kp is not None]
                if valid_pts and cb_dots.isChecked():
                    self.dots.setData(
                        pos=np.array(valid_pts, dtype=float),
                        color=np.array([self.col] * len(valid_pts)))
                    self.dots.setVisible(True)
                else:
                    self.dots.setVisible(False)

                for bi, (a, b) in enumerate(SKEL):
                    pa, pb = kps[a], kps[b]
                    if pa is not None and pb is not None and cb_skel.isChecked():
                        self.bones[bi].setData(
                            pos=np.array([pa, pb], dtype=float))
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
        """Place GL eye AT physical camera position, looking toward room centre."""
        C, R = cam_world_pose(calib, cid)
        if C is None:
            return
        # pyqtgraph spherical camera: eye = center + unit(elev,az) * distance
        # We want eye = C, center = room_center
        # → unit(elev,az) = (C - room_center) / |C - room_center|
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
            vis = monkey_cbs[mi].isChecked()
            obj.visible = vis
            md = fd.get(mid)
            obj.update(
                md["kps"]      if md else None,
                md["com"]      if md else None,
                md["bbox_diag"] if md else 300.0,
            )
            if md and md.get("com"):
                c = md["com"]
                parts.append(f"<b style='color:{MONKEY_COLORS_QT[mi]}'>"
                              f"{mid}</b> ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})")

        info_lbl.setText(
            f"<b>Session:</b> {session_name}<br>"
            f"<b>Folder:</b> {folder.name}<br>"
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

            if cb_2d.isChecked():
                fd_cam = det2d.get(cid, {}).get(frame_num, {})
                for mi, mid in enumerate(monkey_ids):
                    if not monkey_cbs[mi].isChecked():
                        continue
                    det = fd_cam.get(mid)
                    if not det:
                        continue
                    cf = MONKEY_COLORS_F[mi % len(MONKEY_COLORS_F)]
                    bgr = (int(cf[2]*255), int(cf[1]*255), int(cf[0]*255))
                    kps = det["kps"]
                    for a, b in SKEL:
                        pa, pb = kps[a], kps[b]
                        if pa[2] >= KP_CONF_THRESH and pb[2] >= KP_CONF_THRESH:
                            cv2.line(img, (int(pa[0]), int(pa[1])),
                                     (int(pb[0]), int(pb[1])), bgr, 3)
                    for kp in kps:
                        if kp[2] >= KP_CONF_THRESH:
                            cv2.circle(img, (int(kp[0]), int(kp[1])), 5, bgr, -1)

            # cam label burned into frame (top-left)
            cv2.putText(img, f"Cam {cid}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (30, 30, 30), 6)
            col_bgr_cam = tuple(int(c*255) for c in
                                reversed(CAM_COLORS_F[cid][:3]))
            cv2.putText(img, f"Cam {cid}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, col_bgr_cam, 3)

            scale = min(w / VIDEO_W, h / VIDEO_H)
            nw, nh = max(1, int(VIDEO_W * scale)), max(1, int(VIDEO_H * scale))
            img_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
            img_rgb   = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, nw, nh,
                          img_rgb.strides[0], QImage.Format.Format_RGB888)
            pix = QPixmap(w, h)
            pix.fill(QColor(17, 17, 17))
            painter = QPainter(pix)
            painter.drawPixmap((w-nw)//2, (h-nh)//2, QPixmap.fromImage(qimg))
            painter.end()
            lbl.setPixmap(pix)

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
        steps = max(1, int(round(speed)))
        set_frame((cur_idx[0] + steps) % n_frames)

    timer.timeout.connect(tick)

    def toggle_play():
        playing[0] = not playing[0]
        if playing[0]:
            timer.setInterval(max(40, int(200 / speed_combo.currentData())))
            timer.start()
            btn_play.setText("⏸")
        else:
            timer.stop()
            btn_play.setText("▶")

    btn_play.clicked.connect(toggle_play)
    slider.valueChanged.connect(set_frame)
    speed_combo.currentIndexChanged.connect(
        lambda _: timer.setInterval(max(40, int(200/speed_combo.currentData())))
                  if playing[0] else None)

    cb_videos.stateChanged.connect(
        lambda _: (vid_grid_widget.setVisible(cb_videos.isChecked()),
                   update_videos(frame_nums[cur_idx[0]])
                   if cb_videos.isChecked() else None))

    for cb in monkey_cbs + [cb_skel, cb_dots, cb_com, cb_trail, cb_2d]:
        cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))
    trail_combo.currentIndexChanged.connect(lambda _: set_frame(cur_idx[0]))

    # ── Show + initial camera ────────────────────────────────────────────────
    win.show()

    # Set initial GL camera to a top-angled overview of the whole room
    gl_view.opts['center'] = pg.Vector(rx/2, ry/2, rz/4)
    gl_view.setCameraPosition(
        distance=max(rx, ry, rz) * 3.0,
        elevation=30, azimuth=45)

    set_frame(0)
    sys.exit(app.exec())


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

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
