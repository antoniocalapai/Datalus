#!/usr/bin/env python3
"""
MonkeyCalibrate — Four-variant camera self-calibration from monkey-alone sessions.

Calibrations:
  Elmo     — Elmo session only
  Joker    — Joker session only
  Combined — both sessions merged
  Ideal    — both sessions + known camera positions (positions fixed, orientations free)

Scale for Elmo/Joker/Combined: known inter-camera distance of the auto-selected anchor
pair.  No monkey body-size measurement required.

After running, a PyQt6+pyqtgraph comparison window opens.  Click "Save JSON files"
to write the four result JSONs, or "Discard" to exit without saving.

Usage:
    python3 MonkeyCalibrate.py
    python3 MonkeyCalibrate.py --elmo /path/to/Elmo --joker /path/to/Joker
"""

import sys, json, argparse, re
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).parent.parent
HERE    = Path(__file__).parent
OUT_DIR = REPO / "HomeCage_SelfCalibration_Human" / "output"

DEFAULT_SESSIONS = {
    "Elmo":  REPO / "_data" / "MonkeyAlone" / "Elmo _1",
    "Joker": REPO / "_data" / "MonkeyAlone" / "Joker_1",
}
_DEFAULT_INIT_CALIB = OUT_DIR / "calibration_result.json"

# ─── Geometry ─────────────────────────────────────────────────────────────────
CAMERA_IDS = ["102", "108", "113", "117"]

KNOWN_POSITIONS = {
    "102": np.array([300.,  3260., 540.]),
    "108": np.array([1850., 0.,    2480.]),
    "113": np.array([50.,   0.,    550.]),
    "117": np.array([2080., 3070., 2550.]),
}

ROOM = {"x": 2240, "y": 3400, "z": 3260}

# ─── Detection config ─────────────────────────────────────────────────────────
KP_CONF           = 0.30
BBOX_CONF         = 0.30
MIN_QUAL_KPS      = 2           # min visible ANCHOR_KPS per qualifying frame
MIN_ANCHOR_FRAMES = 8           # min simultaneous qualifying frames for anchor pair
ANCHOR_KPS        = [0, 5, 6, 11, 12]   # nose, shoulders, hips
MAX_LANDMARKS_BA  = 2000        # cap landmarks fed into BA (randomly sampled if exceeded)

FRAME_OFFSET = 1_000_000        # shift Joker frame indices when merging sessions

_CAM_PAT = re.compile(r'_(\d{3})_\d+.*_2D_result\.txt$')

# ─── Per-calibration style ────────────────────────────────────────────────────
CALIB_STYLE = {
    "Elmo":     {"hex": "44aaff", "pt": (0.267, 0.667, 1.0,   1.0), "ln": (1.0,   1.0,   0.267, 0.8)},
    "Joker":    {"hex": "ff9144", "pt": (1.0,   0.569, 0.267, 1.0), "ln": (1.0,   0.267, 1.0,   0.8)},
    "Combined": {"hex": "44ff88", "pt": (0.267, 1.0,   0.533, 1.0), "ln": (0.533, 1.0,   0.267, 0.8)},
    "Ideal":    {"hex": "ff44bb", "pt": (1.0,   0.267, 0.733, 1.0), "ln": (1.0,   0.733, 0.267, 0.8)},
    "Human":    {"hex": "ffffff", "pt": (1.0,   1.0,   1.0,   1.0), "ln": (0.8,   0.8,   0.8,   0.8)},
}
CALIB_LABELS = ["Elmo", "Joker", "Combined", "Ideal", "Human"]


# ─── Intrinsics (thin-lens estimate) ─────────────────────────────────────────
def _K(w=2048, h=1496):
    fx = 8.0 * w / 14.16
    return np.array([[fx, 0, w/2.], [0, fx, h/2.], [0, 0, 1.]], dtype=np.float64)


# ─── Data loading ─────────────────────────────────────────────────────────────
def parse_session(folder):
    folder = Path(folder)
    result = {}
    for txt in sorted(folder.glob("*_2D_result.txt")):
        m = _CAM_PAT.search(txt.name)
        if not m:
            continue
        cid = m.group(1)
        cam_data = {}
        with open(txt) as f:
            next(f)  # skip header line
            for line in f:
                parts = line.split()
                if len(parts) < 58:
                    continue
                frame = int(parts[0])
                bbox  = [float(x) for x in parts[2:7]]   # x1 y1 x2 y2 conf
                kps   = [[float(parts[7+i*3]), float(parts[7+i*3+1]),
                          float(parts[7+i*3+2])] for i in range(17)]
                if frame not in cam_data or bbox[4] > cam_data[frame]["bbox"][4]:
                    cam_data[frame] = {"bbox": bbox, "kps": kps}
        result[cid] = cam_data
        print(f"    cam{cid}: {len(cam_data)} frames")
    return result


def merge_sessions(det_a, det_b, offset=FRAME_OFFSET):
    """Merge two session dicts; session-B frame indices are shifted by offset."""
    merged = {}
    for cid in CAMERA_IDS:
        merged[cid] = {}
        if cid in det_a:
            merged[cid].update(det_a[cid])
        if cid in det_b:
            for f, d in det_b[cid].items():
                merged[cid][f + offset] = d
    return merged


# ─── Geometry helpers ─────────────────────────────────────────────────────────
def _qualify(det_cam):
    """Return set of qualifying frame indices for one camera."""
    out = set()
    for f, det in det_cam.items():
        if det["bbox"][4] < BBOX_CONF:
            continue
        kps = np.array(det["kps"])
        if sum(1 for k in ANCHOR_KPS if kps[k, 2] >= KP_CONF) >= MIN_QUAL_KPS:
            out.add(f)
    return out


def _dlt(Ps, pts):
    rows = []
    for P, p in zip(Ps, pts):
        rows += [p[0]*P[2] - P[0], p[1]*P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def _proj(K, R, T, X):
    x = K @ (R @ X + T)
    return x[:2] / x[2]


def _centre(R, T):
    return (-R.T @ T).flatten()


def _procrustes_rot(src, dst):
    """Rotation + translation Procrustes (no scale)."""
    src, dst = np.array(src, dtype=float), np.array(dst, dtype=float)
    sc, dc = src.mean(0), dst.mean(0)
    A = (dst - dc).T @ (src - sc)
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1., 1., np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    return R, dc - R @ sc


def _collect_obs(landmarks, det2d, placed):
    """For each landmark, collect (cam, X3d, p2d) from all placed cameras."""
    obs = []
    for lm in landmarks:
        for _, (f, k) in lm["obs"].items():
            for cam in placed:
                if cam in det2d and f in det2d[cam]:
                    kp = np.array(det2d[cam][f]["kps"])
                    if kp[k, 2] >= KP_CONF:
                        obs.append((cam, lm["X"].copy(), kp[k, :2]))
    return obs


# ─── Calibration A / B / C ────────────────────────────────────────────────────
def calibrate(det2d, label):
    """
    Self-calibration from 2D detections.
    Scale anchored to the known inter-camera distance of the auto-selected anchor pair.
    Returns result dict or None on failure.
    """
    print(f"\n── Calibrate [{label}] ──")
    Ks   = {c: _K() for c in CAMERA_IDS}
    dist = np.zeros(5, dtype=np.float64)

    # Qualify frames per camera
    qual = {}
    for cid in CAMERA_IDS:
        qual[cid] = _qualify(det2d.get(cid, {}))
        print(f"  cam{cid}: {len(qual[cid])} qualifying frames")

    # Select anchor pair (most simultaneous qualifying frames)
    best_pair, best_n = None, 0
    for i, ca in enumerate(CAMERA_IDS):
        for cb in CAMERA_IDS[i+1:]:
            n = len(qual[ca] & qual[cb])
            if n > best_n:
                best_n, best_pair = n, (ca, cb)

    if best_pair is None or best_n < MIN_ANCHOR_FRAMES:
        print(f"  FAIL: best pair has only {best_n} simultaneous frames "
              f"(need ≥ {MIN_ANCHOR_FRAMES})")
        return None

    ca, cb = best_pair
    print(f"  Anchor pair: cam{ca}+cam{cb}  ({best_n} qualifying frames)")

    # Build 2D correspondences for recoverPose
    common = sorted(qual[ca] & qual[cb])
    pts_a, pts_b = [], []
    for f in common:
        if f not in det2d.get(ca, {}) or f not in det2d.get(cb, {}):
            continue
        kp_a = np.array(det2d[ca][f]["kps"])
        kp_b = np.array(det2d[cb][f]["kps"])
        for k in ANCHOR_KPS:
            if kp_a[k, 2] >= KP_CONF and kp_b[k, 2] >= KP_CONF:
                pts_a.append(kp_a[k, :2])
                pts_b.append(kp_b[k, :2])

    pts_a = np.array(pts_a, dtype=np.float64)
    pts_b = np.array(pts_b, dtype=np.float64)
    print(f"  recoverPose correspondences: {len(pts_a)}")
    if len(pts_a) < 10:
        print("  FAIL: too few correspondences")
        return None

    E, _ = cv2.findEssentialMat(pts_a, pts_b, Ks[ca],
                                 method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_rel, T_unit, _ = cv2.recoverPose(E, pts_a, pts_b, Ks[ca])

    # Metric scale from known inter-camera distance
    known_dist = float(np.linalg.norm(KNOWN_POSITIONS[cb] - KNOWN_POSITIONS[ca]))
    T_scaled   = T_unit * known_dist
    print(f"  Scale: dist(cam{ca}↔cam{cb}) = {known_dist:.1f} mm")

    Rs = {ca: np.eye(3),         cb: R_rel}
    Ts = {ca: np.zeros((3, 1)),  cb: T_scaled}

    # Triangulate landmarks from anchor pair
    Pa = Ks[ca] @ np.hstack([Rs[ca], Ts[ca]])
    Pb = Ks[cb] @ np.hstack([Rs[cb], Ts[cb]])

    landmarks = []
    for f in common:
        if f not in det2d.get(ca, {}) or f not in det2d.get(cb, {}):
            continue
        kp_a = np.array(det2d[ca][f]["kps"])
        kp_b = np.array(det2d[cb][f]["kps"])
        for k in ANCHOR_KPS:
            if kp_a[k, 2] >= KP_CONF and kp_b[k, 2] >= KP_CONF:
                X = _dlt([Pa, Pb], [kp_a[k, :2], kp_b[k, :2]])
                if (Rs[ca] @ X + Ts[ca].flatten())[2] > 0 and \
                   (Rs[cb] @ X + Ts[cb].flatten())[2] > 0:
                    landmarks.append({"X": X, "obs": {ca: (f, k), cb: (f, k)}})

    # Filter triangulated landmarks by reprojection error in anchor cameras
    good = []
    for lm in landmarks:
        X = lm["X"]
        errs = [np.linalg.norm(_proj(Ks[c], Rs[c], Ts[c].flatten(), X) - np.array(det2d[c][f]["kps"])[k, :2])
                for c, (f, k) in lm["obs"].items()]
        if max(errs) < 20.0:
            good.append(lm)
    landmarks = good
    print(f"  {len(landmarks)} landmarks after reprojection filter (< 20px)")
    if len(landmarks) < 6:
        print("  FAIL: too few landmarks after filtering")
        return None

    # solvePnP for remaining cameras — use ALL landmarks for maximum coverage
    placed = {ca, cb}
    for cam in [c for c in CAMERA_IDS if c not in placed and c in det2d]:
        pts3, pts2 = [], []
        for lm in landmarks:
            for _, (f, k) in lm["obs"].items():
                if f in det2d.get(cam, {}):
                    kp = np.array(det2d[cam][f]["kps"])
                    if kp[k, 2] >= KP_CONF:
                        pts3.append(lm["X"])
                        pts2.append(kp[k, :2])
                        break
        if len(pts3) < 10:
            print(f"  cam{cam}: only {len(pts3)} pts — skipping solvePnP")
            continue
        pts3 = np.array(pts3, dtype=np.float64)
        pts2 = np.array(pts2, dtype=np.float64)
        ok, rv, tv, inl = cv2.solvePnPRansac(
            pts3, pts2, Ks[cam], dist,
            iterationsCount=2000, reprojectionError=8.0, confidence=0.999)
        if not ok or inl is None or len(inl) < 10:
            print(f"  cam{cam}: solvePnP failed ({len(inl) if inl is not None else 0} inliers)")
            continue
        Rs[cam], _ = cv2.Rodrigues(rv)
        Ts[cam] = tv
        placed.add(cam)
        rp = [np.linalg.norm(_proj(Ks[cam], Rs[cam], Ts[cam].flatten(), pts3[i]) - pts2[i])
              for i in inl.flatten()]
        print(f"  cam{cam}: placed  {len(inl)} inliers  "
              f"reproj={np.median(rp):.1f}px")

    # Subsample landmarks for BA only (solvePnP already done on full set)
    lm_ba = landmarks
    if len(landmarks) > MAX_LANDMARKS_BA:
        step  = len(landmarks) // MAX_LANDMARKS_BA
        lm_ba = landmarks[::step][:MAX_LANDMARKS_BA]
        print(f"  Subsampled to {len(lm_ba)} landmarks for BA")

    # Bundle adjustment — 6 DOF per free camera
    obs  = _collect_obs(lm_ba, det2d, placed)
    # Pre-filter observations with large initial reprojection error
    obs  = [(cam, X, p) for cam, X, p in obs
            if np.linalg.norm(_proj(Ks[cam], Rs[cam], Ts[cam].flatten(), X) - p) < 30.0]
    print(f"  {len(obs)} observations for BA")
    free = [c for c in CAMERA_IDS if c in placed and c != ca]

    def _pack(Rs_d, Ts_d):
        x = []
        for c in free:
            rv, _ = cv2.Rodrigues(Rs_d[c])
            x += rv.flatten().tolist() + Ts_d[c].flatten().tolist()
        return np.array(x, dtype=np.float64)

    def _unpack(x):
        Ru = {ca: Rs[ca]}; Tu = {ca: Ts[ca]}
        for i, c in enumerate(free):
            rv = x[i*6:i*6+3].reshape(3, 1)
            tv = x[i*6+3:i*6+6].reshape(3, 1)
            Ru[c], _ = cv2.Rodrigues(rv)
            Tu[c] = tv
        return Ru, Tu

    def _res(x):
        Ru, Tu = _unpack(x)
        r = []
        for cam, X3d, p2d in obs:
            if cam not in Ru:
                continue
            p = _proj(Ks[cam], Ru[cam], Tu[cam].flatten(), X3d)
            r += (p - p2d).tolist()
        return r

    x0     = _pack(Rs, Ts)
    before = np.median(np.abs(_res(x0)))
    sol    = least_squares(_res, x0, method='trf', loss='soft_l1', max_nfev=3000)
    after  = np.median(np.abs(_res(sol.x)))
    Rs, Ts = _unpack(sol.x)
    print(f"  BA: {before:.1f}px → {after:.1f}px  (median residual)")

    # Similarity Procrustes alignment to world frame (scale + rotation + translation)
    placed_l = [c for c in CAMERA_IDS if c in placed]
    recon    = {c: _centre(Rs[c], Ts[c]) for c in placed_l}
    src_pts  = np.array([recon[c] for c in placed_l])
    dst_pts  = np.array([KNOWN_POSITIONS[c] for c in placed_l])

    # Similarity Procrustes: dst ≈ s * R_al @ src + t_al
    sc, dc   = src_pts.mean(0), dst_pts.mean(0)
    sn, dn   = src_pts - sc, dst_pts - dc
    s_al     = np.linalg.norm(dn) / (np.linalg.norm(sn) + 1e-12)
    U, _, Vt = np.linalg.svd(sn.T @ dn)
    D        = np.diag([1., 1., np.linalg.det(U @ Vt)])
    R_al     = U @ D @ Vt
    t_al     = dc - s_al * R_al @ sc
    aligned  = {c: s_al * R_al @ recon[c] + t_al for c in placed_l}

    errors   = {c: float(np.linalg.norm(aligned[c] - KNOWN_POSITIONS[c]))
                for c in placed_l}
    mean_err = float(np.mean(list(errors.values())))
    print(f"  Procrustes scale: {s_al:.4f}")
    print(f"  Errors:  " +
          "  ".join(f"cam{c}={errors[c]:.0f}mm" for c in placed_l))
    print(f"  Mean: {mean_err:.0f} mm")

    # World-frame R, T  (scale absorbed into T)
    Rs_w = {c: Rs[c] @ R_al.T                                         for c in placed_l}
    Ts_w = {c: Ts[c].flatten() * s_al - Rs[c] @ R_al.T @ t_al        for c in placed_l}

    result = {
        "label":            label,
        "method":           "monkey_pose",
        "anchor_pair":      [ca, cb],
        "scale_mm":         round(known_dist, 1),
        "reproj_before_ba": round(float(before), 2),
        "reproj_after_ba":  round(float(after), 2),
        "mean_error_mm":    round(mean_err, 1),
        "cameras":          {},
    }
    for cid in CAMERA_IDS:
        entry = {"known_mm": KNOWN_POSITIONS[cid].tolist(),
                 "placed":   cid in placed}
        if cid in placed_l:
            a = aligned[cid]
            entry.update({
                "reconstructed_aligned_mm": [round(v, 1) for v in a],
                "error_mm":   round(errors[cid], 1),
                "R_world":    [[round(v, 8) for v in row] for row in Rs_w[cid].tolist()],
                "T_world":    [round(v, 4)  for v in Ts_w[cid].tolist()],
                "K":          [[round(v, 4) for v in row] for row in Ks[cid].tolist()],
            })
        result["cameras"][cid] = entry
    return result


# ─── Ideal calibration (D) ────────────────────────────────────────────────────
def calibrate_ideal(det2d, init_calib_path=None):
    """
    Both sessions merged + camera positions fixed at known values.
    Only camera orientations are optimised (3 DOF each, T derived as T = -R @ C_known).
    Cam 102 is kept at its human-pose R as reference; the other three are free.
    """
    print("\n── Calibrate [Ideal] (known positions + both animals) ──")
    Ks = {c: _K() for c in CAMERA_IDS}

    # Load initial orientations from human-pose calibration
    Rs0 = {}
    if init_calib_path and Path(init_calib_path).exists():
        with open(init_calib_path) as f:
            data = json.load(f)
        for cid in CAMERA_IDS:
            info = data.get("cameras", {}).get(cid, {})
            if "R_world" in info:
                Rs0[cid] = np.array(info["R_world"])
        print(f"  Initial orientations from: {Path(init_calib_path).name}")

    if not Rs0:
        print("  Using look-at initialisation")
        Rs0 = _look_at_init()

    # T such that camera is at known position: T = -R @ C
    Ts0 = {c: (-Rs0[c] @ KNOWN_POSITIONS[c]).reshape(3, 1) for c in Rs0}

    # Triangulate using all camera pairs and the initial calibration
    landmarks = []
    pairs = [(ca, cb) for i, ca in enumerate(CAMERA_IDS)
             for cb in CAMERA_IDS[i+1:]
             if ca in Rs0 and cb in Rs0]
    for ca, cb in pairs:
        Pa = Ks[ca] @ np.hstack([Rs0[ca], Ts0[ca]])
        Pb = Ks[cb] @ np.hstack([Rs0[cb], Ts0[cb]])
        frames_a = {f for f in det2d.get(ca, {})
                    if det2d[ca][f]["bbox"][4] >= BBOX_CONF}
        frames_b = {f for f in det2d.get(cb, {})
                    if det2d[cb][f]["bbox"][4] >= BBOX_CONF}
        for f in sorted(frames_a & frames_b):
            kp_a = np.array(det2d[ca][f]["kps"])
            kp_b = np.array(det2d[cb][f]["kps"])
            for k in ANCHOR_KPS:
                if kp_a[k, 2] >= KP_CONF and kp_b[k, 2] >= KP_CONF:
                    X = _dlt([Pa, Pb], [kp_a[k, :2], kp_b[k, :2]])
                    if (Rs0[ca] @ X + Ts0[ca].flatten())[2] > 0 and \
                       (Rs0[cb] @ X + Ts0[cb].flatten())[2] > 0:
                        landmarks.append({"X": X, "obs": {ca: (f, k), cb: (f, k)}})

    print(f"  Triangulated {len(landmarks)} landmarks from {len(pairs)} pairs")
    if len(landmarks) < 10:
        print("  FAIL: too few landmarks")
        return None

    if len(landmarks) > MAX_LANDMARKS_BA:
        step = len(landmarks) // MAX_LANDMARKS_BA
        landmarks = landmarks[::step][:MAX_LANDMARKS_BA]
        print(f"  Subsampled to {len(landmarks)} landmarks for BA")

    placed = set(Rs0.keys())
    obs    = _collect_obs(landmarks, det2d, placed)
    print(f"  {len(obs)} observations")

    # BA: rvec per camera only; T derived to keep position at known
    ref  = "102"
    free = [c for c in CAMERA_IDS if c in placed and c != ref]

    def _pack_i(Rs_d):
        x = []
        for c in free:
            rv, _ = cv2.Rodrigues(Rs_d[c])
            x += rv.flatten().tolist()
        return np.array(x, dtype=np.float64)

    def _unpack_i(x):
        Ru = {ref: Rs0[ref]}
        Tu = {ref: Ts0[ref]}
        for i, c in enumerate(free):
            rv = x[i*3:(i+1)*3].reshape(3, 1)
            R, _ = cv2.Rodrigues(rv)
            Ru[c] = R
            Tu[c] = (-R @ KNOWN_POSITIONS[c]).reshape(3, 1)
        return Ru, Tu

    def _res_i(x):
        Ru, Tu = _unpack_i(x)
        r = []
        for cam, X3d, p2d in obs:
            if cam not in Ru:
                continue
            p = _proj(Ks[cam], Ru[cam], Tu[cam].flatten(), X3d)
            r += (p - p2d).tolist()
        return r

    x0     = _pack_i(Rs0)
    before = np.median(np.abs(_res_i(x0)))
    sol    = least_squares(_res_i, x0, method='trf', loss='soft_l1', max_nfev=3000)
    after  = np.median(np.abs(_res_i(sol.x)))
    Rs_f, Ts_f = _unpack_i(sol.x)
    print(f"  Ideal BA: {before:.1f}px → {after:.1f}px  (positions fixed)")

    result = {
        "label":            "Ideal",
        "method":           "monkey_pose_ideal",
        "reproj_before_ba": round(float(before), 2),
        "reproj_after_ba":  round(float(after), 2),
        "mean_error_mm":    0.0,
        "cameras":          {},
    }
    for cid in CAMERA_IDS:
        k = KNOWN_POSITIONS[cid]
        entry = {"known_mm": k.tolist(), "placed": cid in placed}
        if cid in placed:
            entry.update({
                "reconstructed_aligned_mm": [round(v, 1) for v in k],
                "error_mm":   0.0,
                "R_world":    [[round(v, 8) for v in row] for row in Rs_f[cid].tolist()],
                "T_world":    [round(float(v), 4) for v in np.array(Ts_f[cid]).flatten()],
                "K":          [[round(v, 4) for v in row] for row in Ks[cid].tolist()],
            })
        result["cameras"][cid] = entry
    return result


def _look_at_init():
    """Fallback: compute initial R by pointing each camera at the room centre."""
    center = np.array([ROOM["x"]/2, ROOM["y"]/2, ROOM["z"]/2])
    Rs = {}
    for cid, C in KNOWN_POSITIONS.items():
        z = center - C;  z /= np.linalg.norm(z)
        up = np.array([0., 0., 1.])
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-6:
            x = np.cross(z, np.array([0., 1., 0.]))
        x /= np.linalg.norm(x)
        y = np.cross(z, x)   # camera Y (down) in world coords
        y /= np.linalg.norm(y)
        # R rows = camera X, Y, Z axes in world frame
        Rs[cid] = np.array([x, y, z])
    return Rs


# ─── PyQt6 comparison window ──────────────────────────────────────────────────
def show_comparison_pyqt(results):
    """Open PyQt6+pyqtgraph 3D comparison window.
    Returns True if user clicks Save, False if Discard/close."""
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QLabel, QPushButton, QCheckBox, QGroupBox,
    )
    from PyQt6.QtCore import Qt
    import pyqtgraph.opengl as gl
    import pyqtgraph as pg

    def _hex_css(hex_str):
        r_, g_, b_ = int(hex_str[0:2],16), int(hex_str[2:4],16), int(hex_str[4:6],16)
        return f"rgb({r_},{g_},{b_})"

    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)

    rx, ry, rz = ROOM["x"], ROOM["y"], ROOM["z"]

    win = QMainWindow()
    win.setWindowTitle("MonkeyCalib — Calibration Comparison")
    win.resize(1400, 820)

    central = QWidget()
    win.setCentralWidget(central)
    root = QHBoxLayout(central)
    root.setContentsMargins(4, 4, 4, 4)
    root.setSpacing(6)

    # ── GL view ───────────────────────────────────────────────────────────────
    gl_view = gl.GLViewWidget()
    gl_view.setBackgroundColor((12, 12, 12, 255))
    gl_view.setCameraPosition(distance=6500, elevation=30, azimuth=45)
    root.addWidget(gl_view, stretch=3)

    # ── Right panel ───────────────────────────────────────────────────────────
    right = QWidget()
    right.setFixedWidth(340)
    right_v = QVBoxLayout(right)
    right_v.setContentsMargins(4, 4, 4, 4)
    right_v.setSpacing(8)
    root.addWidget(right)

    # ── Room wireframe ────────────────────────────────────────────────────────
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

    # ── Known camera positions (large yellow dots) ────────────────────────────
    known_pts = np.array([KNOWN_POSITIONS[c] for c in CAMERA_IDS], dtype=float)
    gl_view.addItem(gl.GLScatterPlotItem(
        pos=known_pts, size=20,
        color=(1.0, 0.9, 0.4, 1.0), pxMode=True))
    # Drop lines to floor
    for pt in known_pts:
        gl_view.addItem(gl.GLLinePlotItem(
            pos=np.array([[pt[0], pt[1], 0], pt], dtype=float),
            color=(1.0, 0.9, 0.4, 0.35), width=1, antialias=True))

    # ── Per-calibration overlays ──────────────────────────────────────────────
    overlay_items = {}
    for label in CALIB_LABELS:
        r = results.get(label)
        if r is None:
            overlay_items[label] = None
            continue
        style = CALIB_STYLE[label]
        pts, line_items = [], []
        for cid in CAMERA_IDS:
            c = r["cameras"].get(cid, {})
            if c.get("placed") and "reconstructed_aligned_mm" in c:
                rv = np.array(c["reconstructed_aligned_mm"], dtype=float)
                kv = np.array(c["known_mm"], dtype=float)
                pts.append(rv)
                li = gl.GLLinePlotItem(
                    pos=np.array([kv, rv], dtype=float),
                    color=style["ln"], width=2, antialias=True)
                gl_view.addItem(li)
                line_items.append(li)
        scatter = None
        if pts:
            scatter = gl.GLScatterPlotItem(
                pos=np.array(pts, dtype=float),
                size=14, color=style["pt"], pxMode=True)
            gl_view.addItem(scatter)
        overlay_items[label] = {"scatter": scatter, "lines": line_items}

    # ── Right: info header ────────────────────────────────────────────────────
    hdr = QLabel(
        f"<b>MonkeyCalib — Calibration Comparison</b><br>"
        f"<span style='color:#888;font-size:10px;'>Room: {rx}×{ry}×{rz} mm  |  "
        f"<span style='color:#ffe066;'>&#9632;</span> Known positions</span>")
    hdr.setStyleSheet("color:#aac8ff; font-size:12px;")
    hdr.setWordWrap(True)
    right_v.addWidget(hdr)

    # ── Toggle checkboxes ─────────────────────────────────────────────────────
    tog_box = QGroupBox("Calibrations")
    tog_v = QVBoxLayout(tog_box)
    tog_v.setSpacing(6)

    for label in CALIB_LABELS:
        r = results.get(label)
        color_css = _hex_css(CALIB_STYLE[label]["hex"])
        cb = QCheckBox(label)
        cb.setChecked(r is not None)
        cb.setEnabled(r is not None)
        cb.setStyleSheet(
            f"color:{color_css}; font-weight:bold; font-size:11px;")
        tog_v.addWidget(cb)

        def _on_toggle(state, lbl=label):
            items = overlay_items.get(lbl)
            if items is None:
                return
            vis = (state == 2)
            if items["scatter"] is not None:
                items["scatter"].setVisible(vis)
            for li in items["lines"]:
                li.setVisible(vis)

        cb.stateChanged.connect(_on_toggle)

    right_v.addWidget(tog_box)

    # ── Stats panel ───────────────────────────────────────────────────────────
    stats_box = QGroupBox("Statistics")
    stats_v = QVBoxLayout(stats_box)
    stats_v.setSpacing(4)

    for label in CALIB_LABELS:
        r = results.get(label)
        if label not in CALIB_STYLE:
            continue
        color_css = _hex_css(CALIB_STYLE[label]["hex"])
        if r is None:
            txt = f"<b style='color:{color_css}'>{label}</b>: not available"
        else:
            reproj = r.get("reproj_after_ba", "?")
            mean_e = r.get("mean_error_mm", 0.0)
            anchor = "+".join(r.get("anchor_pair", []))
            err_str = ("positions fixed" if mean_e == 0.0
                       else f"mean pos error <b>{mean_e:.0f} mm</b>")
            anc_str = f"  anchor=cam{anchor}" if anchor else ""
            txt = (f"<b style='color:{color_css}'>{label}</b>: "
                   f"reproj={reproj}px{anc_str}<br>&nbsp;&nbsp;{err_str}")
        lbl_s = QLabel(txt)
        lbl_s.setStyleSheet("font-size:10px; color:#ccc;")
        lbl_s.setWordWrap(True)
        stats_v.addWidget(lbl_s)

    right_v.addWidget(stats_box)
    right_v.addStretch()

    # ── Save / Discard buttons ────────────────────────────────────────────────
    save_result = [False]

    btn_save = QPushButton("Save JSON files")
    btn_save.setFixedHeight(38)
    btn_save.setStyleSheet(
        "font-size:13px; font-weight:bold; background:#1a4d1a; color:#88ff88;"
        "border:1px solid #44aa44; border-radius:6px;")

    btn_discard = QPushButton("Discard")
    btn_discard.setFixedHeight(38)
    btn_discard.setStyleSheet(
        "font-size:13px; background:#2a1010; color:#ff8888;"
        "border:1px solid #aa4444; border-radius:6px;")

    def _on_save():
        save_result[0] = True
        win.close()

    def _on_discard():
        save_result[0] = False
        win.close()

    btn_save.clicked.connect(_on_save)
    btn_discard.clicked.connect(_on_discard)

    btn_row = QWidget()
    btn_h = QHBoxLayout(btn_row)
    btn_h.setContentsMargins(0, 0, 0, 0)
    btn_h.addWidget(btn_save)
    btn_h.addWidget(btn_discard)
    right_v.addWidget(btn_row)

    win.show()
    win.raise_()
    win.activateWindow()
    app.exec()
    return save_result[0]


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MonkeyCalibrate")
    parser.add_argument("--elmo",    type=Path, default=DEFAULT_SESSIONS["Elmo"],
                        help="Folder with Elmo _2D_result.txt files")
    parser.add_argument("--joker",   type=Path, default=DEFAULT_SESSIONS["Joker"],
                        help="Folder with Joker _2D_result.txt files")
    parser.add_argument("--init-calib", type=Path, default=_DEFAULT_INIT_CALIB,
                        help="calibration_result.json for Ideal BA initialisation")
    parser.add_argument("--view-only", action="store_true",
                        help="Load existing result JSONs and open comparison window (skip calibration)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.view_only:
        # Load previously saved result JSONs directly
        name_map = {
            "Elmo":     "monkey_elmo_result.json",
            "Joker":    "monkey_joker_result.json",
            "Combined": "monkey_combined_result.json",
            "Ideal":    "monkey_ideal_result.json",
            "Human":    "calibration_result.json",
        }
        results = {}
        for label, fname in name_map.items():
            p = OUT_DIR / fname
            if p.exists():
                with open(p) as f:
                    results[label] = json.load(f)
                print(f"  Loaded: {fname}")
            else:
                results[label] = None
                print(f"  Missing: {fname}")
        n_ok = sum(1 for r in results.values() if r is not None)
        print(f"\n{n_ok}/4 results loaded.  Opening comparison window…")
        show_comparison_pyqt(results)
        return

    # Load sessions
    print("Loading Elmo session…")
    det_elmo = parse_session(args.elmo)
    print("Loading Joker session…")
    det_joker = parse_session(args.joker)
    det_combined = merge_sessions(det_elmo, det_joker)

    # Run calibrations
    results = {
        "Elmo":     calibrate(det_elmo,     "Elmo"),
        "Joker":    calibrate(det_joker,    "Joker"),
        "Combined": calibrate(det_combined, "Combined"),
        "Ideal":    calibrate_ideal(det_combined, args.init_calib),
    }

    # Show interactive comparison; user clicks Save or Discard
    n_ok = sum(1 for r in results.values() if r is not None)
    print(f"\n{n_ok}/4 calibrations succeeded.  Opening comparison window…")
    try:
        save_ok = show_comparison_pyqt(results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\nComparison window error: {e}")
        print("Falling back to terminal prompt — press Enter to save, Ctrl-C to discard.")
        try:
            input()
            save_ok = True
        except (KeyboardInterrupt, EOFError):
            save_ok = False

    if not save_ok:
        print("Discarded — no files written.")
        return

    # Save
    name_map = {
        "Elmo":     "monkey_elmo_result.json",
        "Joker":    "monkey_joker_result.json",
        "Combined": "monkey_combined_result.json",
        "Ideal":    "monkey_ideal_result.json",
    }
    for label, r in results.items():
        if r is None:
            print(f"  {label}: failed — skipped")
            continue
        out = OUT_DIR / name_map[label]
        with open(out, "w") as f:
            json.dump(r, f, indent=2)
        print(f"  Saved: {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
