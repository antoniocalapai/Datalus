#!/usr/bin/env python3
"""
HomeCage — Monkey 3D Triangulation & Viewer  (PyQt6 + pyqtgraph + cv2)

Usage:
    python3 monkey_3d.py <folder>  [--calib <calibration_result.json>]

<folder> should contain:
  - ABT 2D inference .txt files  (*_<cam_id>_*_2D_result.txt)
  - raw .mp4 videos              (*_<cam_id>_*.mp4, without keypoints)

--calib  optional path to calibration_result.json.
         If omitted, the script searches upward from <folder> for
         output/calibration_result.json, then falls back to the default
         next to this script.
"""

import sys, json, re, csv, datetime, argparse
import numpy as np
from pathlib import Path

HERE         = Path(__file__).parent
_DEFAULT_CAL = HERE.parent / "HomeCage_SelfCalibration_Human" / "output" / "calibration_result.json"

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

# 17 COCO keypoint names used in the gt_/pred_ CSV files
_CSV_KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]


def parse_csv_2d(path, conf_thresh=0.0):
    """Parse a single GT/RT-prediction CSV → cam_data dict in the same shape
    as parse_2d_results: {frame: {monkey: {bbox, kps}}}.
    Confidence threshold applies per-keypoint (kps below are zeroed)."""
    cam_data = {}
    if not path.exists():
        return cam_data
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame  = int(row["frame_number"])
            monkey = row.get("monkey_name", "monkey")
            try:
                bbox = [float(row["bbox_x1"]), float(row["bbox_y1"]),
                        float(row["bbox_x2"]), float(row["bbox_y2"]),
                        float(row.get("bbox_confidence") or 1.0)]
            except (ValueError, KeyError):
                continue
            kps = []
            ok = False
            for name in _CSV_KP_NAMES:
                sx = row.get(f"{name}_x", "")
                sy = row.get(f"{name}_y", "")
                sc = row.get(f"{name}_confidence", "")
                if sx.strip() and sy.strip():
                    c = float(sc) if sc.strip() else 1.0
                    if c >= conf_thresh:
                        kps.append([float(sx), float(sy), c])
                        ok = True
                    else:
                        kps.append([0.0, 0.0, 0.0])
                else:
                    kps.append([0.0, 0.0, 0.0])
            if not ok:
                continue
            if frame not in cam_data:
                cam_data[frame] = {}
            if (monkey not in cam_data[frame] or
                bbox[4] > cam_data[frame][monkey]["bbox"][4]):
                cam_data[frame][monkey] = {"bbox": bbox, "kps": kps}
    return cam_data


def parse_csv_session(csv_dir, prefix, session, conf_thresh=0.0):
    """Load 4-camera CSV session into the standard det2d shape:
       {cam_id: {frame: {monkey: {bbox, kps}}}}."""
    out = {}
    for cid in CAM_ORDER:
        path = Path(csv_dir) / f"{prefix}_{session}_cam{cid}.csv"
        if path.exists():
            d = parse_csv_2d(path, conf_thresh=conf_thresh)
            if d:
                out[cid] = d
                print(f"  cam{cid} ({prefix}): {sum(len(v) for v in d.values())} dets "
                      f"across {len(d)} frames")
    return out


def find_videos_session(videos_root, session):
    """Find eval_cam<id>.mp4 files inside videos_root/<session>/."""
    out = {}
    sess_dir = Path(videos_root) / session
    if not sess_dir.exists():
        return out
    for cid in CAM_ORDER:
        for name in (f"eval_cam{cid}.mp4", f"cam{cid}.mp4"):
            p = sess_dir / name
            if p.exists():
                out[cid] = p
                break
    return out


# ─── Calibration ──────────────────────────────────────────────────────────────
def find_calib(folder, override=None):
    """Locate calibration_result.json: override > parent search > default."""
    if override:
        p = Path(override)
        if p.exists():
            return p
        raise FileNotFoundError(f"--calib not found: {override}")
    # Walk upward from folder looking for output/calibration_result.json
    for parent in [folder] + list(folder.parents):
        candidate = parent / "output" / "calibration_result.json"
        if candidate.exists():
            return candidate
    # Final fallback: sibling of this script
    if _DEFAULT_CAL.exists():
        return _DEFAULT_CAL
    raise FileNotFoundError(
        "calibration_result.json not found. Pass --calib <path>.")

def load_calib(path):
    with open(path) as f:
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
    """Find raw (non-annotated) videos.

    Strategy:
    1. For each *_2D_result.txt, strip '_2D_result' to get the raw video name.
    2. Fallback: any *.mp4 whose name contains '_<cam_id>_' and is NOT a
       _2D_result.mp4.
    3. Last resort: _2D_result.mp4 (annotated) if nothing else found.
    """
    folder = Path(folder)
    videos = {}

    # Strategy 1: raw video adjacent to each txt file
    for txt in sorted(folder.glob("*_2D_result.txt")):
        m = _CAM_PAT.search(txt.name)
        if not m:
            continue
        cid = m.group(1)
        raw_name = txt.stem.replace("_2D_result", "") + ".mp4"
        raw_path = folder / raw_name
        if raw_path.exists():
            videos[cid] = raw_path

    if videos:
        return videos

    # Strategy 2: any non-annotated mp4 matching _<cam_id>_
    _cid_pat = re.compile(r'_(\d{3})[_\.]')
    for vid in sorted(folder.glob("*.mp4")):
        if "_2D_result" in vid.name:
            continue
        m = _cid_pat.search(vid.name)
        if m and m.group(1) in CAM_ORDER and m.group(1) not in videos:
            videos[m.group(1)] = vid

    if videos:
        return videos

    # Strategy 3: annotated fallback
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
def merge_gt_pred_det2d(det2d_gt, det2d_pred):
    """Merge two source dicts into one det2d, suffixing monkey ids with
    _GT / _RT so the existing pipeline draws them as separate animals."""
    merged = {}
    cams = set(det2d_gt) | set(det2d_pred)
    for cam in cams:
        merged[cam] = {}
        for src, suffix in [(det2d_gt.get(cam, {}), "_GT"),
                            (det2d_pred.get(cam, {}), "_RT")]:
            for f, mdict in src.items():
                if f not in merged[cam]:
                    merged[cam][f] = {}
                for m, det in mdict.items():
                    merged[cam][f][m + suffix] = det
    return merged


def run_viewer(frames_out, det2d, videos, calib, session_name, folder,
               extra_calibs=None, gt_vs_pred=False, all_sessions=None):
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

    # ── Color helpers (used by both ticks bar and right-panel widgets) ───────
    GTRT_COLORS = {
        "Elm": {"GT": (0.20, 1.00, 1.00, 1.0),
                "RT": (0.30, 0.50, 1.00, 1.0)},
        "Jok": {"GT": (1.00, 0.95, 0.20, 1.0),
                "RT": (1.00, 0.45, 0.15, 1.0)},
    }
    def gtrt_color(mid):
        for prefix, cols in GTRT_COLORS.items():
            if mid.startswith(prefix + "_") or mid == prefix:
                if mid.endswith("_GT"): return cols["GT"]
                if mid.endswith("_RT"): return cols["RT"]
        return None
    def color_for(mid, mi):
        if gt_vs_pred:
            c = gtrt_color(mid)
            if c is not None:
                r, g, b = (int(round(v * 255)) for v in c[:3])
                return f"#{r:02x}{g:02x}{b:02x}"
        return MONKEY_COLORS_QT[mi % len(MONKEY_COLORS_QT)]

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
    cam_pos_lines = "<br>".join(
        f"  {cid}: ({CAM_KNOWN_POS[cid][0]:.0f}, "
        f"{CAM_KNOWN_POS[cid][1]:.0f}, {CAM_KNOWN_POS[cid][2]:.0f}) mm"
        for cid in CAM_ORDER)
    info_lbl.setText(
        f"<b>Session:</b> {session_name}<br>"
        f"<b>Room:</b> {rx}×{ry}×{rz} mm<br>"
        f"<b>Cameras:</b><br>{cam_pos_lines}<br>"
        f"<b>Subjects:</b> {', '.join(monkey_ids)}<br>"
        f"<b>Frames:</b> {n_frames}  ·  5 fps"
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

    # ── Per-source presence rows for the ticks bar ──────────────────────────
    # In gt_vs_pred mode, build one row per (animal, source) showing where
    # that combination has a 3D reconstruction. Otherwise fall back to the
    # legacy "any reconstruction" green bar.
    def _compute_tick_rows():
        rows = []   # list of (label, color, set_of_frame_indices)
        if gt_vs_pred:
            for mid in monkey_ids:
                idxs = set()
                for fi, fn in enumerate(frame_nums):
                    fd = frames_out.get(fn, {})
                    if mid in fd:
                        idxs.add(fi)
                hex_color = color_for(mid, 0)
                rows.append((mid, QColor(hex_color), idxs))
            # Per-animal GT∩RT overlap rows (white)
            animals = sorted({mid[:-3] for mid in monkey_ids
                              if mid.endswith("_GT") or mid.endswith("_RT")})
            for animal in animals:
                gt_id, rt_id = animal + "_GT", animal + "_RT"
                idxs = set()
                for fi, fn in enumerate(frame_nums):
                    fd = frames_out.get(fn, {})
                    if gt_id in fd and rt_id in fd:
                        idxs.add(fi)
                rows.append((f"{animal} GT∩RT", QColor(255, 255, 255, 230), idxs))
        else:
            rows.append(("recon", QColor(60, 210, 60, 210), set(good_frame_indices)))
        return rows

    tick_rows = [_compute_tick_rows()]   # mutable holder

    # ── GT vs RT error stats ─────────────────────────────────────────────────
    def _compute_gtrt_stats():
        """Return dict {animal: {n_overlap_frames, all_distances_mm}}."""
        stats = {}
        if not gt_vs_pred:
            return stats
        animals = sorted({mid[:-3] for mid in monkey_ids
                          if mid.endswith("_GT") or mid.endswith("_RT")})
        for animal in animals:
            gt_id = animal + "_GT"
            rt_id = animal + "_RT"
            n_overlap = 0
            dists = []
            for fn in frame_nums:
                fd = frames_out.get(fn, {})
                gt = fd.get(gt_id)
                rt = fd.get(rt_id)
                if not gt or not rt:
                    continue
                kg = gt.get("kps") or []
                kr = rt.get("kps") or []
                pair_count = 0
                for a, b in zip(kg, kr):
                    if a is not None and b is not None:
                        d = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
                        dists.append(d)
                        pair_count += 1
                if pair_count > 0:
                    n_overlap += 1
            stats[animal] = {"n_overlap": n_overlap, "dists": np.array(dists)}
        return stats

    gtrt_stats = [_compute_gtrt_stats()]

    def _format_gtrt_summary():
        if not gt_vs_pred:
            return ""
        s = gtrt_stats[0]
        if not s:
            return "no data"
        lines = ["<b>Aggregate (overlap frames)</b>"]
        for animal, d in s.items():
            arr = d["dists"]
            if arr.size == 0:
                lines.append(f"<b>{animal}</b>  no overlap")
                continue
            lines.append(
                f"<b>{animal}</b>  n={d['n_overlap']}f  "
                f"med={np.median(arr):.0f}  "
                f"mean={arr.mean():.0f}  "
                f"p90={np.percentile(arr,90):.0f} mm")
        return "<br>".join(lines)

    # Slider container: ticks bar stacked above slider
    ROW_H = 4
    class FrameTicksBar(QWidget):
        def __init__(self):
            super().__init__()
            n_rows = max(1, len(tick_rows[0]))
            self.setFixedHeight(ROW_H * n_rows + 2)
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.setToolTip("Detection rows (per animal × source)")

        def refresh(self):
            n_rows = max(1, len(tick_rows[0]))
            self.setFixedHeight(ROW_H * n_rows + 2)
            self.update()

        def paintEvent(self, _e):
            p = QPainter(self)
            p.fillRect(self.rect(), QColor(18, 18, 18))
            if n_frames <= 1:
                return
            w = max(1, self.width() - 1)
            for ri, (label, color, idxs) in enumerate(tick_rows[0]):
                y = ri * ROW_H + 1
                for idx in idxs:
                    x = int(idx / (n_frames - 1) * w)
                    p.fillRect(x, y, 2, ROW_H - 1, color)

        def mousePressEvent(self, e):
            if n_frames <= 1:
                return
            raw = round(e.pos().x() / max(1, self.width()-1) * (n_frames-1))
            raw = max(0, min(n_frames-1, raw))
            # Snap to nearest frame that has ANY detection in current rows
            all_idxs = set().union(*(r[2] for r in tick_rows[0])) if tick_rows[0] else set()
            target = min(all_idxs, key=lambda i: abs(i-raw)) if all_idxs else raw
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

    btn_open = QPushButton("📂")
    btn_open.setFixedSize(34, 28)
    btn_open.setToolTip("Open session folder")
    btn_open.setStyleSheet("font-size:13px;padding:0;")

    ctrl_h.addWidget(btn_play)
    ctrl_h.addWidget(slider_container, stretch=1)
    ctrl_h.addWidget(lbl_frame)
    ctrl_h.addWidget(speed_combo)
    ctrl_h.addWidget(btn_rec)
    ctrl_h.addWidget(btn_open)

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
    cb_videos = make_cb("Video feeds", checked=not gt_vs_pred)

    trail_combo = QComboBox()
    for lbl_t, val in [("30 f", 30), ("60 f", 60), ("2 min", 120), ("5 min", 300), ("∞", 99999)]:
        trail_combo.addItem(lbl_t, val)
    trail_combo.setCurrentIndex(1)

    trail_row = QWidget()
    tr_h = QHBoxLayout(trail_row)
    tr_h.setContentsMargins(0, 0, 0, 0)
    tr_h.addWidget(cb_trail); tr_h.addWidget(trail_combo); tr_h.addStretch()


    monkey_cbs = []
    monkey_cbs_container = QWidget()
    monkey_cbs_vbox = QVBoxLayout(monkey_cbs_container)
    monkey_cbs_vbox.setContentsMargins(0, 0, 0, 0)
    monkey_cbs_vbox.setSpacing(2)
    for mi, mid in enumerate(monkey_ids):
        color = color_for(mid, mi)
        cb = make_cb(f"● {mid}")
        cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
        monkey_cbs.append(cb)
        monkey_cbs_vbox.addWidget(cb)

    mode_combo = QComboBox()
    mode_combo.addItem("Mode A — Reconstructed  (internally consistent)", 0)
    mode_combo.addItem("Mode B — Physical  (Procrustes-corrected)", 1)
    mode_combo.setCurrentIndex(1)
    mode_combo.setStyleSheet("font-size:10px;")

    # Calibration dropdown — re-triangulates the session on change
    calib_combo = QComboBox()
    calib_combo.setStyleSheet("font-size:10px;")
    if extra_calibs:
        for label in extra_calibs:
            calib_combo.addItem(label, label)
    else:
        calib_combo.addItem("(default)", None)
        calib_combo.setEnabled(False)

    # Session dropdown — swaps the entire session (det2d + videos)
    session_combo = QComboBox()
    session_combo.setStyleSheet("font-size:10px;")
    if all_sessions:
        for sess_name in all_sessions:
            session_combo.addItem(sess_name, sess_name)
        session_combo.setCurrentText(session_name.split()[0])
    else:
        session_combo.addItem("(single)", None)
        session_combo.setEnabled(False)

    tog_v.addWidget(QLabel("Session:"))
    tog_v.addWidget(session_combo)
    tog_v.addWidget(QLabel("Calibration:"))
    tog_v.addWidget(calib_combo)
    for w in [mode_combo, cb_skel, cb_dots, cb_com, trail_row]:
        tog_v.addWidget(w)
    tog_v.addWidget(QLabel("── Monkeys ──"))
    tog_v.addWidget(monkey_cbs_container)
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

    # ── GT vs RT error panel ─────────────────────────────────────────────────
    err_box = QGroupBox("GT vs RT error")
    err_v   = QVBoxLayout(err_box)
    err_v.setSpacing(2)
    err_v.setContentsMargins(6, 6, 6, 6)
    err_lbl = QLabel("—")
    err_lbl.setStyleSheet("font:10px monospace; color:#ddd;")
    err_lbl.setWordWrap(True)
    err_lbl.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    err_v.addWidget(err_lbl)
    err_v.addStretch()
    if not gt_vs_pred:
        err_box.setVisible(False)

    # top bar of the right panel: Display | Best sequences | Error
    top_controls = QWidget()
    top_ctrl_h   = QHBoxLayout(top_controls)
    top_ctrl_h.setContentsMargins(0, 0, 0, 0)
    top_ctrl_h.setSpacing(4)
    top_ctrl_h.addWidget(tog_box, stretch=3)
    top_ctrl_h.addWidget(seq_box, stretch=2)
    if gt_vs_pred:
        top_ctrl_h.addWidget(err_box, stretch=2)
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
        lbl.setStyleSheet("background:#111; border:2px solid #444;")
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lbl.setMinimumSize(200, 150)
        lbl.setText(f"Cam {cid}")
        vid_grid.addWidget(lbl, row, col)
        vid_labels[cid] = lbl

    right_v.addWidget(vid_grid_widget, stretch=1)
    vid_grid_widget.setVisible(cb_videos.isChecked())
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
            self.mid = mid
            c = gtrt_color(mid) if gt_vs_pred else None
            self.col = c if c is not None else MONKEY_COLORS_F[mi % len(MONKEY_COLORS_F)]
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

        def destroy(self):
            for b in self.bones:
                gl_view.removeItem(b)
            gl_view.removeItem(self.dots)
            gl_view.removeItem(self.com_item)
            gl_view.removeItem(self.trail_line)

    monkey_objs = [MonkeyObjs(mi, mid) for mi, mid in enumerate(monkey_ids)]

    # ── Camera snap ──────────────────────────────────────────────────────────
    room_center   = np.array([rx/2, ry/2, rz/4])
    active_snap   = [None]   # currently hidden camera cone

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
        gl_view.setCameraPosition(distance=dist + 300, elevation=elev, azimuth=azimuth)
        # hide this camera's cone, restore previous
        if active_snap[0] and active_snap[0] != cid:
            cam_cone_items[active_snap[0]].setVisible(True)
            if active_snap[0] in vid_labels:
                vid_labels[active_snap[0]].setStyleSheet(
                    "background:#111; border:2px solid #444;")
        if cid in cam_cone_items:
            cam_cone_items[cid].setVisible(False)
        if cid in vid_labels:
            vid_labels[cid].setStyleSheet(
                "background:#111; border:3px solid #ff6600;")
        active_snap[0] = cid
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
                parts.append(f"<b style='color:{color_for(mid, mi)}'>"
                              f"{mid}</b> ({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})")

        mode_str = "A · Reconstructed" if frame_mode[0] == 0 else "B · Physical"
        info_lbl.setText(
            f"<b>Session:</b> {session_name}<br>"
            f"<b>Mode:</b> {mode_str}<br>"
            f"<b>Room:</b> {rx}×{ry}×{rz} mm<br>"
            f"<b>Frame:</b> #{frame_num}  ({idx+1}/{n_frames})  ·  5 fps<br>"
            + ("<br>".join(parts) if parts else "<i>no detection</i>")
        )
        info_lbl.adjustSize()

        if cb_videos.isChecked():
            update_videos(frame_num)

        # GT vs RT live error display
        if gt_vs_pred:
            agg = _format_gtrt_summary()
            live_lines = []
            animals = sorted({mid[:-3] for mid in monkey_ids
                              if mid.endswith("_GT") or mid.endswith("_RT")})
            for animal in animals:
                gt = fd.get(animal + "_GT")
                rt = fd.get(animal + "_RT")
                if not gt or not rt:
                    continue
                kg = gt.get("kps") or []
                kr = rt.get("kps") or []
                ds = []
                for a, b in zip(kg, kr):
                    if a is not None and b is not None:
                        ds.append(float(np.linalg.norm(np.asarray(a) - np.asarray(b))))
                if ds:
                    arr = np.asarray(ds)
                    live_lines.append(
                        f"<b>{animal}</b> n={len(ds)}  "
                        f"med={np.median(arr):.0f}  max={arr.max():.0f} mm")
            live_html = ("<b>Frame</b><br>" + "<br>".join(live_lines)
                         if live_lines else "<b>Frame</b><br><i>no overlap</i>")
            err_lbl.setText(live_html + "<br><br>" + agg)

    # Per-animal BGR colours for the 2D overlays. Falls back to a generic
    # palette if the animal name isn't a known monkey.
    _OVERLAY_BGR = {
        "Elm":    (178, 114,   0),    # Wong blue, BGR
        "Jok":    (  0, 159, 230),    # Wong orange, BGR
        "Elm_GT": (178, 114,   0),
        "Elm_RT": (178, 114,   0),
        "Jok_GT": (  0, 159, 230),
        "Jok_RT": (  0, 159, 230),
        "Human":  (230, 200, 120),    # light blue, BGR
    }
    _OVERLAY_FALLBACK = (230, 200, 120)   # light blue default

    def _draw_2d_overlay(img, dets):
        """dets: dict monkey -> {"bbox": [x1,y1,x2,y2,conf?], "kps": [[x,y,c]]*17}"""
        for monkey, d in (dets or {}).items():
            color = _OVERLAY_BGR.get(monkey, _OVERLAY_FALLBACK)
            bbox = d.get("bbox") or []
            if len(bbox) >= 4:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox[:4])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
                label = monkey
                ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                cv2.rectangle(img, (x1, y1 - ts[1] - 14),
                              (x1 + ts[0] + 12, y1), color, -1)
                cv2.putText(img, label, (x1 + 6, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            kps = d.get("kps") or []
            for a, b in SKEL:
                if a < len(kps) and b < len(kps):
                    ka, kb = kps[a], kps[b]
                    if len(ka) >= 3 and len(kb) >= 3 and ka[2] > 0.3 and kb[2] > 0.3:
                        cv2.line(img, (int(ka[0]), int(ka[1])),
                                      (int(kb[0]), int(kb[1])), color, 3)
            for kp in kps:
                if len(kp) >= 3 and kp[2] > 0.3:
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 5, (255,255,255), -1)
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 5, color, 2)

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
            # 2D detection overlay (RT bbox + 17 keypoints + skeleton)
            cam_data = det2d.get(cid, {}) if det2d else {}
            _draw_2d_overlay(img, cam_data.get(frame_num, {}))
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

    def on_calib_change(_=None):
        if not extra_calibs:
            return
        nonlocal frames_out, calib, cam_pos_recon, cam_pos_physical, cam_R
        nonlocal procrustes_params
        label = calib_combo.currentData()
        new_calib = extra_calibs.get(label)
        if new_calib is None:
            return
        calib = new_calib
        Ps = build_projection_matrices(calib)
        frames_out = triangulate_session(det2d, Ps)
        # refresh frame index in case it shrank
        new_frame_nums = sorted(frames_out.keys())
        if not new_frame_nums:
            return
        # mutate the closure-shared frame_nums in place
        frame_nums.clear()
        frame_nums.extend(new_frame_nums)
        slider.blockSignals(True)
        slider.setRange(0, len(frame_nums) - 1)
        slider.blockSignals(False)
        # rebuild camera geometry
        cam_pos_recon.clear(); cam_pos_physical.clear(); cam_R.clear()
        for cid, info in calib["cameras"].items():
            if "reconstructed_aligned_mm" in info:
                cam_pos_recon[cid] = list(info["reconstructed_aligned_mm"])
            if "known_mm" in info:
                cam_pos_physical[cid] = list(info["known_mm"])
            if "R_world" in info:
                cam_R[cid] = np.array(info["R_world"])
        _src = [cam_pos_recon[c] for c in CAM_ORDER
                if c in cam_pos_recon and c in cam_pos_physical]
        _dst = [cam_pos_physical[c] for c in CAM_ORDER
                if c in cam_pos_recon and c in cam_pos_physical]
        procrustes_params = compute_procrustes(_src, _dst) if len(_src) >= 3 else None
        update_cam_markers()
        for obj in monkey_objs:
            obj.trail_pts = []
        tick_rows[0] = _compute_tick_rows()
        ticks_bar.refresh()
        gtrt_stats[0] = _compute_gtrt_stats()
        set_frame(min(cur_idx[0], len(frame_nums) - 1))
        print(f"Switched calibration → {label}")

    calib_combo.currentIndexChanged.connect(on_calib_change)

    def on_session_change(_=None):
        if not all_sessions:
            return
        nonlocal det2d, videos, frames_out, monkey_objs, monkey_cbs, monkey_ids
        nonlocal n_frames, caps
        sess_name = session_combo.currentData()
        sess = all_sessions.get(sess_name)
        if sess is None:
            return
        det2d   = sess["det2d"]
        videos  = sess["videos"]
        # Reopen video captures
        for cap in caps.values():
            cap.release()
        caps.clear()
        import cv2 as _cv2
        for cid in CAM_ORDER:
            if cid in videos:
                cap = _cv2.VideoCapture(str(videos[cid]))
                if cap.isOpened():
                    caps[cid] = cap
        # Re-triangulate with current calibration
        Ps = build_projection_matrices(calib)
        frames_out = triangulate_session(det2d, Ps)
        new_frame_nums = sorted(frames_out.keys())
        frame_nums.clear()
        frame_nums.extend(new_frame_nums)
        n_frames = len(frame_nums)
        slider.blockSignals(True)
        slider.setRange(0, max(0, n_frames - 1))
        slider.setValue(0)
        slider.blockSignals(False)
        cur_idx[0] = 0
        # Rebuild monkey IDs / objects (sessions may have different animals)
        new_monkey_ids = sorted({m for fd in frames_out.values() for m in fd})
        for obj in monkey_objs:
            obj.destroy()
        monkey_ids.clear()
        monkey_ids.extend(new_monkey_ids)
        # Rebuild checkboxes
        _clear_layout(monkey_cbs_vbox)
        monkey_cbs.clear()
        for mi, mid in enumerate(monkey_ids):
            color = color_for(mid, mi)
            cb = make_cb(f"● {mid}")
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
            monkey_cbs.append(cb)
            monkey_cbs_vbox.addWidget(cb)
            cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))
        monkey_objs.clear()
        monkey_objs.extend(MonkeyObjs(mi, mid) for mi, mid in enumerate(monkey_ids))
        tick_rows[0] = _compute_tick_rows()
        ticks_bar.refresh()
        gtrt_stats[0] = _compute_gtrt_stats()
        win.setWindowTitle(f"HomeCage 3D · {sess_name} (GT vs RT)  ({folder.name})")
        set_frame(0)
        print(f"Switched session → {sess_name}  ({n_frames} frames)")

    session_combo.currentIndexChanged.connect(on_session_change)

    for cb in monkey_cbs + [cb_skel, cb_dots, cb_com, cb_trail]:
        cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))
    trail_combo.currentIndexChanged.connect(lambda _: set_frame(cur_idx[0]))

    # ── Session reload ───────────────────────────────────────────────────────
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def rebuild_seq_buttons():
        _clear_layout(seq_v)
        lbl_b = QLabel("● Both animals, ≥2 cams")
        lbl_b.setStyleSheet("font-size:9px; color:#88ccff; padding:2px 0;")
        seq_v.addWidget(lbl_b)
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
        lbl_s = QLabel("● Single animal, ≥3 cams")
        lbl_s.setStyleSheet("font-size:9px; color:#ffcc44; padding:2px 0;")
        seq_v.addWidget(lbl_s)
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

    def do_reload(new_folder):
        nonlocal frames_out, det2d, videos, frame_nums, monkey_ids, n_frames
        nonlocal good_frame_indices, top5_runs, top5_single_runs, caps
        nonlocal session_name, folder, monkey_objs, monkey_cbs
        nonlocal cam_pos_recon, cam_pos_physical, cam_R, procrustes_params, calib

        from PyQt6.QtWidgets import QMessageBox
        if playing[0]:
            toggle_play()

        # ── Load calibration ──────────────────────────────────────────────
        try:
            calib_path = find_calib(new_folder)
        except FileNotFoundError as e:
            QMessageBox.critical(win, "Calibration not found", str(e))
            return
        calib = load_calib(calib_path)
        Ps    = build_projection_matrices(calib)

        # ── Load session data ─────────────────────────────────────────────
        videos    = find_videos(new_folder)
        det2d     = parse_2d_results(new_folder)
        frames_out = triangulate_session(det2d, Ps)

        folder = new_folder
        if folder.name.lower() in ("2d_results", "results", "inference"):
            session_name = folder.parent.name
        else:
            session_name = folder.name

        # ── Camera geometry ───────────────────────────────────────────────
        cam_pos_recon.clear(); cam_pos_physical.clear(); cam_R.clear()
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

        # ── Session stats ─────────────────────────────────────────────────
        frame_nums = sorted(frames_out.keys())
        monkey_ids = sorted({m for fd in frames_out.values() for m in fd})
        n_frames   = len(frame_nums)

        good_frame_indices = set()
        _path_mm  = {mid: 0.0 for mid in monkey_ids}
        _prev_com = {mid: None for mid in monkey_ids}
        _min_prox = float('inf')
        for fi, fn in enumerate(frame_nums):
            fd   = frames_out.get(fn, {})
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
                if _prev_com[mid] is not None:
                    _path_mm[mid] += float(np.linalg.norm(com - _prev_com[mid]))
                _prev_com[mid] = com
            if len(monkey_ids) >= 2 and len(coms) == 2:
                mids = list(coms.keys())
                d = float(np.linalg.norm(coms[mids[0]] - coms[mids[1]]))
                if d < _min_prox: _min_prox = d

        _path_strs = ", ".join(
            f"<b style='color:{MONKEY_COLORS_QT[i]}'>{mid}</b> {_path_mm[mid]/1000:.1f}m"
            for i, mid in enumerate(monkey_ids))
        _prox_str = f"{_min_prox/10:.0f} cm" if _min_prox < float('inf') else "—"
        stats_lbl.setText(
            f"Path: {_path_strs}  ·  Min proximity: {_prox_str}  ·  "
            f"Reconstructed: {len(good_frame_indices)}/{n_frames} frames")

        # ── Top-5 runs ────────────────────────────────────────────────────
        good_sorted = sorted(good_frame_indices)
        top5_runs   = []
        if good_sorted:
            rs, rl = good_sorted[0], 1
            for i in range(1, len(good_sorted)):
                if frame_nums[good_sorted[i]] == frame_nums[good_sorted[i-1]] + 1:
                    rl += 1
                else:
                    top5_runs.append((rl, rs, good_sorted[i-1]))
                    rs, rl = good_sorted[i], 1
            top5_runs.append((rl, rs, good_sorted[-1]))
            top5_runs.sort(reverse=True)
            top5_runs = top5_runs[:5]

        _cam_count = {}
        for cam_id, cam_data in det2d.items():
            for fn, fd in cam_data.items():
                if fn not in _cam_count: _cam_count[fn] = {}
                for mid in fd: _cam_count[fn][mid] = _cam_count[fn].get(mid, 0) + 1
        _single_good = set()
        for fi, fn in enumerate(frame_nums):
            if any(c >= 3 for c in _cam_count.get(fn, {}).values()):
                _single_good.add(fi)
        single_sorted   = sorted(_single_good)
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

        # ── Release old caps, open new ────────────────────────────────────
        for cap in caps.values(): cap.release()
        caps.clear()
        for cid in CAM_ORDER:
            if cid in videos:
                cap = cv2.VideoCapture(str(videos[cid]))
                if cap.isOpened(): caps[cid] = cap

        # ── Remove old monkey GL objects ──────────────────────────────────
        for obj in monkey_objs:
            obj.destroy()

        # ── Rebuild monkey checkboxes ─────────────────────────────────────
        _clear_layout(monkey_cbs_vbox)
        monkey_cbs = []
        for mi, mid in enumerate(monkey_ids):
            color = MONKEY_COLORS_QT[mi % len(MONKEY_COLORS_QT)]
            cb = make_cb(f"● {mid}")
            cb.setStyleSheet(f"QCheckBox {{ color: {color}; }}")
            monkey_cbs.append(cb)
            monkey_cbs_vbox.addWidget(cb)
            cb.stateChanged.connect(lambda _: set_frame(cur_idx[0]))

        # ── Rebuild monkey 3D objects ─────────────────────────────────────
        monkey_objs = [MonkeyObjs(mi, mid) for mi, mid in enumerate(monkey_ids)]

        # ── Update camera markers ─────────────────────────────────────────
        update_cam_markers()

        # ── Rebuild best-sequence buttons ─────────────────────────────────
        rebuild_seq_buttons()

        # ── Update slider + ticks ─────────────────────────────────────────
        slider.blockSignals(True)
        slider.setRange(0, max(0, n_frames - 1))
        slider.setValue(0)
        slider.blockSignals(False)
        cur_idx[0] = 0
        ticks_bar.update()

        # ── Update window title + go to frame 0 ──────────────────────────
        win.setWindowTitle(f"HomeCage 3D · {session_name}  ({folder.name})")
        set_frame(0)
        print(f"Reloaded session: {session_name}  ({folder})")

    def open_session():
        from PyQt6.QtWidgets import QFileDialog
        new_folder = QFileDialog.getExistingDirectory(
            win, "Open session folder", str(folder))
        if new_folder:
            do_reload(Path(new_folder).resolve())

    btn_open.clicked.connect(open_session)

    # ── Keyboard shortcuts (QShortcut — works regardless of focus) ───────────
    from PyQt6.QtGui import QKeySequence, QShortcut

    QShortcut(QKeySequence(Qt.Key.Key_Space), win).activated.connect(toggle_play)
    QShortcut(QKeySequence(Qt.Key.Key_1),     win).activated.connect(lambda: snap_camera(CAM_ORDER[0]))
    QShortcut(QKeySequence(Qt.Key.Key_2),     win).activated.connect(lambda: snap_camera(CAM_ORDER[1]))
    QShortcut(QKeySequence(Qt.Key.Key_3),     win).activated.connect(lambda: snap_camera(CAM_ORDER[2]))
    QShortcut(QKeySequence(Qt.Key.Key_4),     win).activated.connect(lambda: snap_camera(CAM_ORDER[3]))

    # ── Show ─────────────────────────────────────────────────────────────────
    win.show()
    gl_view.opts['center'] = pg.Vector(rx/2, ry/2, rz/4)
    gl_view.setCameraPosition(distance=max(rx, ry, rz)*3.0, elevation=30, azimuth=45)
    set_frame(0)
    sys.exit(app.exec())


# ─── Main ─────────────────────────────────────────────────────────────────────
_VAL_SESSIONS = ["250708"]

def _gt_vs_pred_main(initial_session):
    """Launch the viewer in GT-vs-RT comparison mode.
    Pre-loads all validation sessions so the user can switch via dropdown."""
    val_root = (Path(__file__).parent.parent /
                "_data/3d_validation/260402_HomeCage_702D_ElmJok")
    gt_dir   = val_root / "gt"
    pred_dir = val_root / "RT_predictions"
    vid_root = val_root / "videos"
    cal_dir  = (Path(__file__).parent.parent /
                "HomeCage_SelfCalibration_Human/output")

    print(f"\n── GT vs RT mode (multi-session) ──")
    all_sessions = {}
    for sess in _VAL_SESSIONS:
        print(f"\n── {sess} ──")
        det_gt   = parse_csv_session(gt_dir,   "gt",   sess, conf_thresh=0.0)
        det_pred = parse_csv_session(pred_dir, "pred", sess, conf_thresh=0.30)
        if not det_gt and not det_pred:
            print("  no data, skipping")
            continue
        det2d  = merge_gt_pred_det2d(det_gt, det_pred)
        videos = find_videos_session(vid_root, sess)
        all_sessions[sess] = {"det2d": det2d, "videos": videos}
        print(f"  videos: " + ", ".join(f"cam{c}" for c in sorted(videos)))
    if not all_sessions:
        print("No sessions loaded.")
        sys.exit(1)

    print("\n── Loading calibrations ──")
    keep = ("calibration_result", "monkey_gt_")
    label_override = {"calibration_result": "Human Pose"}
    calibs = {}
    for path in sorted(cal_dir.glob("*_result.json")):
        if not any(path.stem.startswith(k) or path.stem == k for k in keep):
            continue
        try:
            d = json.loads(path.read_text())
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            continue
        label = (label_override.get(path.stem)
                 or d.get("label")
                 or path.stem.replace("_result", "").replace("_", " ").title())
        calibs[label] = d
        print(f"  {label}")
    if not calibs:
        print("No calibrations found.")
        sys.exit(1)

    first_label = next(iter(calibs))
    calib = calibs[first_label]
    Ps    = build_projection_matrices(calib)

    sess0 = initial_session if initial_session in all_sessions else next(iter(all_sessions))
    det2d  = all_sessions[sess0]["det2d"]
    videos = all_sessions[sess0]["videos"]

    print(f"\n── Triangulating session {sess0} with {first_label} ──")
    frames_out = triangulate_session(det2d, Ps)
    print(f"  frames with 3D: {len(frames_out)}")

    folder = vid_root / sess0
    run_viewer(frames_out, det2d, videos, calib,
               session_name=f"{sess0} (GT vs RT)",
               folder=folder, extra_calibs=calibs, gt_vs_pred=True,
               all_sessions=all_sessions)


# ─── Human calibration viewer ─────────────────────────────────────────────────
_HUMAN_CALIB_POSE_DIR = Path("/Users/acalapai/PycharmProjects/Datalus/HomeCage_SelfCalibration_Human/output")
_HUMAN_CALIB_VIDEO_DIR = {
    "250707": Path("/Users/acalapai/PycharmProjects/Datalus/_data/CalibrationVideos/250707"),
    "250708": Path("/Users/acalapai/PycharmProjects/Datalus/_data/CalibrationVideos/250708"),
}
_HUMAN_CALIB_VIDEO_PATTERNS = {
    "250707": "Calibration_4_{cam}_20250707154928.mp4",
    "250708": "_2_{cam}_20250708161657.mp4",
}


def _parse_human_pose_file(path):
    """Parse pose_<cam>_<session>.txt (pairs of lines: header + 51 kp values).
    Returns det2d shape: {frame: {monkey: {"bbox": [x1,y1,x2,y2,conf], "kps": [[x,y,c]×17]}}}"""
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines) - 1:
        head = lines[i].split()
        kp_line = lines[i + 1].split()
        i += 2
        try:
            frame = int(head[0])
            bbox = [float(head[1]), float(head[2]),
                    float(head[3]), float(head[4]),
                    float(head[5]) if len(head) > 5 else 1.0]
        except (ValueError, IndexError):
            continue
        kps = []
        for k in range(17):
            try:
                x = float(kp_line[k*3])
                y = float(kp_line[k*3 + 1])
                c = float(kp_line[k*3 + 2])
                kps.append([x, y, c])
            except (ValueError, IndexError):
                kps.append([0.0, 0.0, 0.0])
        out[frame] = {"Human": {"bbox": bbox, "kps": kps}}
    return out


def _human_calib_main(session):
    """Launch the viewer on the human calibration walk for the HomeCage."""
    print(f"\n── Human calibration viewer · session {session} ──")

    calib_path = _PAPER_CAL
    if not calib_path.exists():
        print(f"Calibration not found: {calib_path}")
        sys.exit(1)
    calib = json.loads(calib_path.read_text())
    Ps = build_projection_matrices(calib)
    print(f"  cameras placed: {sorted(Ps)}")

    print("\n── Loading YOLO26x-pose detections ──")
    det2d = {}
    for cid in CAM_ORDER:
        path = _HUMAN_CALIB_POSE_DIR / f"pose_{cid}_{session}.txt"
        d = _parse_human_pose_file(path)
        if d:
            det2d[cid] = d
            print(f"  cam{cid}: {len(d)} frames")

    if not det2d:
        print("No human pose files found"); sys.exit(1)

    print("\n── Locating calibration videos ──")
    videos = {}
    vid_dir = _HUMAN_CALIB_VIDEO_DIR.get(session)
    pattern = _HUMAN_CALIB_VIDEO_PATTERNS.get(session, "")
    if vid_dir and vid_dir.exists():
        for cid in CAM_ORDER:
            vname = pattern.replace("{cam}", cid)
            vpath = vid_dir / vname
            if vpath.exists():
                videos[cid] = vpath
                print(f"  cam{cid}: {vname}")
            else:
                print(f"  cam{cid}: not found ({vname})")

    print("\n── Triangulating ──")
    frames_out = triangulate_session(det2d, Ps)
    print(f"  frames with 3D output: {len(frames_out)}")

    folder = vid_dir or _HUMAN_CALIB_POSE_DIR
    run_viewer(frames_out, det2d, videos, calib,
               session_name=f"Human calib {session}",
               folder=folder)


# ─── HomeCagePaper integration ───────────────────────────────────────────────
# Map session id → list of candidate raw-video locations.
_PAPER_VIDEO_LOCATIONS = {
    "250711": Path("/Users/acalapai/PycharmProjects/Datalus/_data/sessions/250711/RAW"),
    "250713": Path("/Users/acalapai/PycharmProjects/Datalus/_data/videos/250713"),
    "250715": Path("/Users/acalapai/PycharmProjects/Datalus/_data/3d_validation/260402_HomeCage_702D_ElmJok/videos/250715"),
}
_PAPER_DET_DIR = Path("/Users/acalapai/PycharmProjects/Datalus/HomeCagePaper/data/sessions")
_PAPER_CAL     = Path("/Users/acalapai/PycharmProjects/Datalus/HomeCagePaper/data/calibrations/human_pose.json")


def _parse_paper_session_dets(session, conf_thresh=0.30):
    """Read HomeCagePaper RT detection CSVs (detections_camNNN.txt)."""
    sess_dir = _PAPER_DET_DIR / session
    out = {}
    for cid in CAM_ORDER:
        path = sess_dir / f"detections_cam{cid}.txt"
        if path.exists():
            d = parse_csv_2d(path, conf_thresh=conf_thresh)
            if d:
                out[cid] = d
                print(f"  cam{cid}: {sum(len(v) for v in d.values())} dets "
                      f"across {len(d)} frames")
    return out


def _find_paper_videos(session):
    """Look up the raw mp4s for a HomeCagePaper session."""
    root = _PAPER_VIDEO_LOCATIONS.get(session)
    out = {}
    if root is None or not root.exists():
        return out
    for cid in CAM_ORDER:
        for pattern in (f"*_{cid}_*.mp4", f"*cam{cid}*.mp4"):
            matches = sorted(root.glob(pattern))
            if matches:
                out[cid] = matches[0]
                break
    return out


def _paper_session_main(session):
    """Launch the viewer on a HomeCagePaper session, using the Human Pose
    calibration and the RT detections from HomeCagePaper/data/sessions/."""
    print(f"\n── HomeCagePaper session {session} ──")

    if not _PAPER_CAL.exists():
        print(f"Calibration not found: {_PAPER_CAL}")
        sys.exit(1)
    calib = json.loads(_PAPER_CAL.read_text())
    Ps    = build_projection_matrices(calib)
    print(f"  cameras placed: {sorted(Ps)}")

    print("\n── Loading RT detections ──")
    det2d = _parse_paper_session_dets(session)
    if not det2d:
        print(f"No detections under {_PAPER_DET_DIR / session}")
        sys.exit(1)

    print("\n── Locating raw videos ──")
    videos = _find_paper_videos(session)
    for cid in CAM_ORDER:
        v = videos.get(cid)
        print(f"  cam{cid}: {v.name if v else '— missing —'}")
    if not videos:
        print(f"No videos found for {session}. Edit "
              f"_PAPER_VIDEO_LOCATIONS in monkey_3d.py if needed.")

    print("\n── Triangulating ──")
    frames_out = triangulate_session(det2d, Ps)
    print(f"  frames with 3D output: {len(frames_out)}")

    folder = (_PAPER_VIDEO_LOCATIONS.get(session)
              or (_PAPER_DET_DIR / session))
    run_viewer(frames_out, det2d, videos, calib,
               session_name=f"HCpaper {session}",
               folder=folder)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HomeCage 3D viewer")
    ap.add_argument("folder", nargs="?",
                    help="Folder with ABT txt files and raw videos")
    ap.add_argument("--calib", default=None,
                    help="Path to calibration_result.json (auto-detected if omitted)")
    ap.add_argument("--gt-vs-pred", metavar="SESSION", default=None,
                    help="Compare GT vs RT predictions for a validation session "
                         "(e.g. 250708). Loads CSVs from "
                         "_data/3d_validation/260402_HomeCage_702D_ElmJok/")
    ap.add_argument("--human-calib", metavar="SESSION", default=None,
                    help="View human calibration walk (e.g. 250708). "
                         "Loads YOLO26x-pose detections + calibration videos.")
    ap.add_argument("--paper-session", metavar="SESSION", default=None,
                    help="Open a HomeCagePaper session by id (e.g. 250713). "
                         "Reads RT detections from HomeCagePaper/data/sessions/<SESSION>/, "
                         "videos from the configured raw-video locations.")
    args = ap.parse_args()

    if args.gt_vs_pred:
        _gt_vs_pred_main(args.gt_vs_pred)
        sys.exit(0)

    if args.human_calib:
        _human_calib_main(args.human_calib)
        sys.exit(0)

    if args.paper_session:
        _paper_session_main(args.paper_session)
        sys.exit(0)

    if not args.folder:
        ap.error("either provide a folder, --gt-vs-pred SESSION, or --paper-session SESSION")

    folder = Path(args.folder).resolve()
    if not folder.exists():
        print(f"ERROR: {folder} does not exist"); sys.exit(1)

    # Derive session name: use grandparent if folder is named 2D_results/,
    # otherwise use the folder name itself
    if folder.name.lower() in ("2d_results", "results", "inference"):
        session_name = folder.parent.name
    else:
        session_name = folder.name
    print(f"Session: {session_name}  ({folder})")

    print("\n── Calibration ──")
    try:
        calib_path = find_calib(folder, args.calib)
    except FileNotFoundError as e:
        print(f"ERROR: {e}"); sys.exit(1)
    print(f"  Using: {calib_path}")
    calib = load_calib(calib_path)
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
