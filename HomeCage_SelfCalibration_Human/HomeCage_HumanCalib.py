#!/usr/bin/env python3
"""
HomeCage Human Self-Calibration
Stage 1: Inspect    — read video metadata
Stage 2: Extract    — sample frames at 2000ms intervals
Stage 3: Detect     — YOLO26x-pose keypoints
Stage 4: Preview    — HTML with sample frames + skeleton overlay
Stage 5: Calibrate  — self-calibration (recoverPose → scale → solvePnP → BA)
                      compares reconstructed camera positions to known ground truth
"""

import cv2, base64, json, os, multiprocessing as mp
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from ultralytics import YOLO

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO  = Path("/Users/acalapai/PycharmProjects/Datalus")
HERE  = Path(__file__).parent
OUT   = HERE / "output"
MODEL = REPO / "_models" / "yolo26x-pose.pt"

SESSIONS = {
    "250707": {
        "102": REPO / "_data/CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4",
        "108": REPO / "_data/CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4",
        "113": REPO / "_data/CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4",
        "117": REPO / "_data/CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4",
    },
    "250708": {
        "102": REPO / "_data/CalibrationVideos/250708/_2_102_20250708161657.mp4",
        "108": REPO / "_data/CalibrationVideos/250708/_2_108_20250708161657.mp4",
        "113": REPO / "_data/CalibrationVideos/250708/_2_113_20250708161657.mp4",
        "117": REPO / "_data/CalibrationVideos/250708/_2_117_20250708161657.mp4",
    },
}
CAMERA_IDS = ["102", "108", "113", "117"]

# ─── Config ───────────────────────────────────────────────────────────────────
FRAME_INTERVAL_MS  = 2000  # sample 1 frame per 2000ms (preview run; use 500 for full calib)
BBOX_CONF          = 0.30
KP_CONF            = 0.50
N_PREVIEW          = 8    # sample frames shown per camera per session in preview

# ─── Calibration config ───────────────────────────────────────────────────────
CALIB_SESSION    = "250708"   # session to use for calibration
PERSON_HEIGHT_MM = 1700       # update to actual walker height before running
SCALE_BBOX_CONF  = 0.70       # min bbox confidence for scale/anchor frames
SCALE_VERT_FRAC  = 0.30       # min vertical body span (fraction of image height)
SCALE_KPS        = [0, 5, 6, 11, 12, 15, 16]  # nose, shoulders, hips, ankles

# ─── Known camera positions (ground truth, mm) ────────────────────────────────
KNOWN_POSITIONS = {
    "102": np.array([300.,  3260., 540.]),
    "108": np.array([1850., 0.,    2480.]),
    "113": np.array([50.,   0.,    550.]),
    "117": np.array([2080., 3070., 2550.]),
}

# COCO 17 skeleton pairs
COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Inspect
# ─────────────────────────────────────────────────────────────────────────────
def stage1_inspect():
    print("\n── Stage 1: Inspect ──")
    meta = {}
    for sess, cams in SESSIONS.items():
        meta[sess] = {}
        for cam, path in cams.items():
            cap = cv2.VideoCapture(str(path))
            w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            meta[sess][cam] = {"w": w, "h": h, "fps": fps, "n_frames": n}
            print(f"  {sess}/cam{cam}: {w}x{h} @ {fps:.1f}fps  {n} frames ({n/fps:.0f}s)")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Extract frames
# ─────────────────────────────────────────────────────────────────────────────
def stage2_extract(meta):
    print("\n── Stage 2: Extract frames ──")
    frame_map = {}   # sess -> cam -> [(frame_idx, Path), ...]

    for sess, cams in SESSIONS.items():
        frame_map[sess] = {}
        for cam, path in cams.items():
            fps      = meta[sess][cam]["fps"]
            n_frames = meta[sess][cam]["n_frames"]
            step     = max(1, round(fps * FRAME_INTERVAL_MS / 1000.0))
            indices  = list(range(0, n_frames, step))

            out_dir = OUT / "frames" / sess / cam
            out_dir.mkdir(parents=True, exist_ok=True)

            # resume: if exactly the expected number of PNGs exist, skip
            existing = sorted(out_dir.glob("*.png"))
            if len(existing) == len(indices):
                print(f"  {sess}/cam{cam}: {len(existing)} frames already on disk — skip")
                frame_map[sess][cam] = list(zip(indices, existing))
                continue

            cap = cv2.VideoCapture(str(path))
            entries = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                img_path = out_dir / f"{cam}_{sess}_{idx:06d}.png"
                cv2.imwrite(str(img_path), frame)
                entries.append((idx, img_path))
            cap.release()
            frame_map[sess][cam] = entries
            print(f"  {sess}/cam{cam}: extracted {len(entries)} frames  (step={step})")

    return frame_map


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Pose detection
# ─────────────────────────────────────────────────────────────────────────────
def _load_pose_file(pose_path):
    detections = []
    with open(pose_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i + 1 < len(lines):
        t1 = lines[i].split()
        t2 = lines[i+1].split()
        fidx = int(t1[0])
        bbox = list(map(float, t1[1:]))          # x1 y1 x2 y2 conf
        kps  = list(map(float, t2))              # 51 floats: kx ky kc × 17
        detections.append((fidx, bbox, kps))
        i += 2
    return detections


def stage3_detect(frame_map):
    print("\n── Stage 3: Pose detection (YOLO26x-pose) ──")
    model = YOLO(str(MODEL))

    pose_data = {}   # sess -> cam -> [(fidx, bbox_5, kps_51), ...]

    for sess in SESSIONS:
        pose_data[sess] = {}
        for cam in CAMERA_IDS:
            pose_path = OUT / f"pose_{cam}_{sess}.txt"
            frames    = frame_map[sess][cam]

            if pose_path.exists():
                pose_data[sess][cam] = _load_pose_file(pose_path)
                n = len(pose_data[sess][cam])
                print(f"  {sess}/cam{cam}: loaded {n} detections from {pose_path.name}")
                continue

            detections = []
            with open(pose_path, "w") as f:
                for k, (fidx, img_path) in enumerate(frames):
                    if k % 50 == 0:
                        print(f"    {sess}/cam{cam}: {k}/{len(frames)}...", end="\r")
                    results = model(str(img_path), conf=BBOX_CONF, verbose=False)
                    r = results[0]
                    if r.keypoints is None or len(r.boxes) == 0:
                        continue
                    best = int(r.boxes.conf.argmax())
                    box  = r.boxes.xyxy[best].cpu().numpy().tolist()
                    conf = float(r.boxes.conf[best])
                    kps  = r.keypoints.data[best].cpu().numpy().flatten().tolist()

                    f.write(f"{fidx} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f} {conf:.4f}\n")
                    f.write(" ".join(f"{v:.2f}" for v in kps) + "\n")
                    detections.append((fidx, box + [conf], kps))

            pose_data[sess][cam] = detections
            pct = 100 * len(detections) / max(1, len(frames))
            print(f"  {sess}/cam{cam}: {len(detections)}/{len(frames)} detections ({pct:.1f}%)       ")

    return pose_data


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Preview HTML
# ─────────────────────────────────────────────────────────────────────────────
def _draw_pose(img, bbox, kps_flat):
    out = img.copy()
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    kps = np.array(kps_flat).reshape(17, 3)
    pts = {}
    for i, (kx, ky, kc) in enumerate(kps):
        if kc >= KP_CONF:
            pts[i] = (int(kx), int(ky))
            cv2.circle(out, (int(kx), int(ky)), 6, (0, 200, 255), -1)
    for a, b in COCO_PAIRS:
        if a in pts and b in pts:
            cv2.line(out, pts[a], pts[b], (255, 120, 0), 2)
    return out


def _to_b64(img, max_w=480):
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(h * max_w / w)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
    return base64.b64encode(buf.tobytes()).decode()


def stage4_preview(frame_map, pose_data):
    print("\n── Stage 4: Build preview HTML ──")

    sections = []   # list of (session_label, rows)
                    # rows = list of (cam_label, stats, [b64, ...])

    for sess in ["250707", "250708"]:
        rows = []
        for cam in CAMERA_IDS:
            dets   = pose_data[sess][cam]
            frames = frame_map[sess][cam]
            n_det  = len(dets)
            n_tot  = len(frames)
            stats  = f"{n_det} / {n_tot} frames detected  ({100*n_det/max(1,n_tot):.1f}%)"

            idx2path = {fidx: p for fidx, p in frames}
            step = max(1, n_det // N_PREVIEW)
            samples = dets[::step][:N_PREVIEW]

            imgs = []
            for fidx, bbox, kps in samples:
                p = idx2path.get(fidx)
                if p is None:
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                drawn = _draw_pose(img, bbox, kps)
                imgs.append(_to_b64(drawn))

            rows.append((f"Cam {cam}", stats, imgs))
            print(f"  {sess}/cam{cam}: {stats}  →  {len(imgs)} preview frames")

        sections.append((f"Session {sess}", rows))

    html = _render_html(sections)
    out_path = OUT / "preview.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\n  Saved: {out_path}")
    return out_path


def _render_html(sections):
    body = ""
    for sess_label, rows in sections:
        body += f'<h2>{sess_label}</h2>\n'
        for cam_label, stats, imgs in rows:
            imgs_html = "".join(
                f'<img src="data:image/jpeg;base64,{b}" '
                f'style="width:230px;margin:3px;border-radius:3px;border:1px solid #333;">'
                for b in imgs
            ) or '<p style="color:#666;margin:0;">No detections</p>'
            body += f"""
<div style="margin-bottom:20px;">
  <div style="font-weight:bold;color:#cde;margin-bottom:3px;">{cam_label}</div>
  <div style="font-size:12px;color:#888;margin-bottom:6px;">{stats}</div>
  <div style="display:flex;flex-wrap:wrap;">{imgs_html}</div>
</div>
"""
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>HomeCage Human Calibration — Pose Preview</title>
<style>
  body{{background:#181818;color:#ccc;font-family:monospace;padding:24px;margin:0;}}
  h1{{color:#fff;border-bottom:1px solid #333;padding-bottom:10px;margin-bottom:6px;}}
  h2{{color:#8cf;margin-top:28px;margin-bottom:12px;border-left:3px solid #8cf;padding-left:10px;}}
  p.sub{{color:#666;margin:0 0 20px;font-size:13px;}}
</style>
</head><body>
<h1>HomeCage Human Self-Calibration — Pose Detection Preview</h1>
<p class="sub">YOLO26x-pose &nbsp;·&nbsp; COCO 17-point &nbsp;·&nbsp;
bbox_conf &ge; {BBOX_CONF} &nbsp;·&nbsp; kp_conf &ge; {KP_CONF}</p>
{body}
</body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 — Self-calibration + ground-truth comparison
# ─────────────────────────────────────────────────────────────────────────────

def _intrinsics(cam_id, img_w, img_h):
    """Thin-lens estimate. Sensor width 14.16mm = Sony IMX304 1.1" format at full res
    (4096px), 2x2 binned to 2048px. Lens: VS-Technology V0828-MPY2, focal length 8mm."""
    fx = 8.0 * img_w / 14.16
    K  = np.array([[fx, 0, img_w/2.],
                   [0, fx, img_h/2.],
                   [0,  0,       1.]], dtype=np.float64)
    return K, np.zeros(5, dtype=np.float64)


def _qualifying_frames(dets, img_h):
    """Return set of frame indices where the detection is a full-body pose."""
    out = set()
    for fidx, bbox, kps in dets:
        if bbox[4] < SCALE_BBOX_CONF:
            continue
        kp = np.array(kps).reshape(17, 3)
        if not all(kp[k, 2] >= KP_CONF for k in SCALE_KPS):
            continue
        ys = [kp[k, 1] for k in SCALE_KPS]
        if (max(ys) - min(ys)) < SCALE_VERT_FRAC * img_h:
            continue
        out.add(fidx)
    return out


def _dlt_triangulate(P1, P2, p1, p2):
    """Triangulate one point from two projection matrices and 2D points."""
    A = np.array([
        p1[0]*P1[2] - P1[0],
        p1[1]*P1[2] - P1[1],
        p2[0]*P2[2] - P2[0],
        p2[1]*P2[2] - P2[1],
    ])
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


def _project(K, R, T, X3d):
    """Project 3D point X3d (3,) to 2D pixel with K, R, T."""
    x = K @ (R @ X3d + T)
    return x[:2] / x[2]


def _cam_center(R, T):
    """Camera centre in world coords from OpenCV R, T."""
    return (-R.T @ T).flatten()


def _procrustes_rotation(source, target):
    """
    Rotation-only Procrustes: find R, t such that R @ source_i + t ≈ target_i.
    source, target: (N, 3) arrays.
    Returns R (3×3), t (3,).
    """
    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    A = (target - tgt_c).T @ (source - src_c)
    U, _, Vt = np.linalg.svd(A)
    # enforce proper rotation (det=+1)
    D = np.diag([1, 1, np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    t = tgt_c - R @ src_c
    return R, t


def stage5_calibrate(pose_data, meta):
    print(f"\n── Stage 5: Self-calibration (session {CALIB_SESSION}) ──")
    sess  = CALIB_SESSION
    dets  = pose_data[sess]
    img_h = meta[sess]["102"]["h"]   # used for vertical span filter (same for all cams)

    # ── 5.1  Intrinsics ──────────────────────────────────────────────────────
    Ks, dists = {}, {}
    for cam in CAMERA_IDS:
        w = meta[sess][cam]["w"]
        h = meta[sess][cam]["h"]
        Ks[cam], dists[cam] = _intrinsics(cam, w, h)
        print(f"  cam{cam}: K (thin-lens) fx={Ks[cam][0,0]:.1f}px")

    # ── 5.2  Anchor pair selection ───────────────────────────────────────────
    qual = {cam: _qualifying_frames(dets[cam], img_h) for cam in CAMERA_IDS}
    for cam in CAMERA_IDS:
        print(f"  cam{cam}: {len(qual[cam])} qualifying full-body frames")

    best_pair, best_n = None, 0
    for i, ca in enumerate(CAMERA_IDS):
        for cb in CAMERA_IDS[i+1:]:
            n = len(qual[ca] & qual[cb])
            print(f"  pair ({ca},{cb}): {n} common qualifying frames")
            if n > best_n:
                best_n, best_pair = n, (ca, cb)

    cam_a, cam_b = best_pair
    print(f"\n  Anchor pair: cam{cam_a} + cam{cam_b}  ({best_n} frames)")

    # ── 5.3  Build 2D-2D correspondences for anchor pair ────────────────────
    idx2det = {cam: {fidx: (bbox, kps) for fidx, bbox, kps in dets[cam]}
               for cam in CAMERA_IDS}

    common_frames = sorted(qual[cam_a] & qual[cam_b])
    pts_a, pts_b  = [], []
    for fidx in common_frames:
        if fidx not in idx2det[cam_a] or fidx not in idx2det[cam_b]:
            continue
        kp_a = np.array(idx2det[cam_a][fidx][1]).reshape(17, 3)
        kp_b = np.array(idx2det[cam_b][fidx][1]).reshape(17, 3)
        for k in SCALE_KPS:
            if kp_a[k, 2] >= KP_CONF and kp_b[k, 2] >= KP_CONF:
                pts_a.append(kp_a[k, :2])
                pts_b.append(kp_b[k, :2])

    pts_a = np.array(pts_a, dtype=np.float64)
    pts_b = np.array(pts_b, dtype=np.float64)
    print(f"  Correspondences for recoverPose: {len(pts_a)}")

    # ── 5.4  recoverPose ─────────────────────────────────────────────────────
    E, mask = cv2.findEssentialMat(pts_a, pts_b, Ks[cam_a],
                                   method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R_rel, T_unit, mask_rp = cv2.recoverPose(E, pts_a, pts_b, Ks[cam_a])

    # cam_a: R=I, T=0;  cam_b: R_rel, T_unit (unit scale)
    Rs  = {cam_a: np.eye(3),  cam_b: R_rel}
    Ts  = {cam_a: np.zeros((3,1)), cam_b: T_unit}

    # ── 5.5  Metric scale from person height ─────────────────────────────────
    P_a = Ks[cam_a] @ np.hstack([Rs[cam_a], Ts[cam_a]])
    P_b = Ks[cam_b] @ np.hstack([Rs[cam_b], Ts[cam_b]])

    scales = []
    for fidx in common_frames:
        if fidx not in idx2det[cam_a] or fidx not in idx2det[cam_b]:
            continue
        kp_a = np.array(idx2det[cam_a][fidx][1]).reshape(17, 3)
        kp_b = np.array(idx2det[cam_b][fidx][1]).reshape(17, 3)
        # head: kp0  |  ankles: kp15, kp16
        head_kps  = [0]
        ankle_kps = [15, 16]
        for h in head_kps:
            for ak in ankle_kps:
                if (kp_a[h,2] >= KP_CONF and kp_b[h,2] >= KP_CONF and
                    kp_a[ak,2] >= KP_CONF and kp_b[ak,2] >= KP_CONF):
                    X_head  = _dlt_triangulate(P_a, P_b, kp_a[h,:2],  kp_b[h,:2])
                    X_ankle = _dlt_triangulate(P_a, P_b, kp_a[ak,:2], kp_b[ak,:2])
                    # cheirality: both must have positive depth in cam_a
                    if (Rs[cam_a] @ X_head  + Ts[cam_a].flatten())[2] > 0 and \
                       (Rs[cam_a] @ X_ankle + Ts[cam_a].flatten())[2] > 0:
                        scales.append(np.linalg.norm(X_head - X_ankle))

    if not scales:
        print("  ERROR: no valid head-ankle triangulations for scale")
        return None

    scale = PERSON_HEIGHT_MM / np.median(scales)
    Ts[cam_b] = Ts[cam_b] * scale
    print(f"  Scale factor: {scale:.4f}  (from {len(scales)} head-ankle pairs)")
    print(f"  |cam_a→cam_b| = {np.linalg.norm(Ts[cam_b]):.1f} mm")

    # ── 5.6  Triangulate SCALE_KPS as metric 3D landmarks ───────────────────
    P_a = Ks[cam_a] @ np.hstack([Rs[cam_a], Ts[cam_a]])
    P_b = Ks[cam_b] @ np.hstack([Rs[cam_b], Ts[cam_b]])

    landmarks_3d = []  # list of (X3d, [(cam, kp_idx, fidx), ...])
    for fidx in common_frames:
        if fidx not in idx2det[cam_a] or fidx not in idx2det[cam_b]:
            continue
        kp_a = np.array(idx2det[cam_a][fidx][1]).reshape(17, 3)
        kp_b = np.array(idx2det[cam_b][fidx][1]).reshape(17, 3)
        for k in SCALE_KPS:
            if kp_a[k,2] >= KP_CONF and kp_b[k,2] >= KP_CONF:
                X = _dlt_triangulate(P_a, P_b, kp_a[k,:2], kp_b[k,:2])
                # cheirality filter
                if (Rs[cam_a] @ X + Ts[cam_a].flatten())[2] > 0 and \
                   (Rs[cam_b] @ X + Ts[cam_b].flatten())[2] > 0:
                    landmarks_3d.append({
                        "X": X,
                        "obs": {cam_a: (fidx, k), cam_b: (fidx, k)}
                    })

    print(f"  Triangulated {len(landmarks_3d)} landmarks from anchor pair")

    # ── 5.7  solvePnP for remaining cameras ──────────────────────────────────
    remaining = [c for c in CAMERA_IDS if c not in (cam_a, cam_b)]
    placed    = {cam_a, cam_b}

    for cam in remaining:
        # build 3D-2D correspondences
        pts3d, pts2d = [], []
        for lm in landmarks_3d:
            for src_cam, (fidx, k) in lm["obs"].items():
                if fidx in idx2det[cam]:
                    kp = np.array(idx2det[cam][fidx][1]).reshape(17, 3)
                    if kp[k, 2] >= KP_CONF:
                        pts3d.append(lm["X"])
                        pts2d.append(kp[k, :2])
                        break

        if len(pts3d) < 6:
            print(f"  cam{cam}: only {len(pts3d)} correspondences — skipping solvePnP")
            continue

        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, Ks[cam], dists[cam],
            iterationsCount=2000, reprojectionError=8.0, confidence=0.999
        )
        if not ok or inliers is None or len(inliers) < 4:
            print(f"  cam{cam}: solvePnP failed ({len(inliers) if inliers is not None else 0} inliers)")
            continue

        Rs[cam], _ = cv2.Rodrigues(rvec)
        Ts[cam]    = tvec
        placed.add(cam)

        # report reprojection error
        reproj = []
        for i in inliers.flatten():
            p2 = _project(Ks[cam], Rs[cam], Ts[cam].flatten(), pts3d[i])
            reproj.append(np.linalg.norm(p2 - pts2d[i]))
        print(f"  cam{cam}: solvePnP OK  {len(inliers)} inliers  "
              f"reproj={np.median(reproj):.1f}px")

    # ── 5.8  Bundle adjustment ───────────────────────────────────────────────
    print(f"\n  Bundle adjustment over {len(placed)} cameras…")

    # Collect all observations: (cam, X3d, p2d)
    # For each landmark, check every placed camera for a matching detection.
    obs = []
    for lm in landmarks_3d:
        for src_cam, (fidx, k) in lm["obs"].items():
            for cam in placed:
                if fidx in idx2det[cam]:
                    kp = np.array(idx2det[cam][fidx][1]).reshape(17, 3)
                    if kp[k, 2] >= KP_CONF:
                        obs.append((cam, lm["X"].copy(), kp[k, :2]))

    # Free cameras: all except cam_a (fixed at R=I, T=0)
    free_cams = [c for c in placed if c != cam_a]

    def _pack(Rs_d, Ts_d):
        x = []
        for c in free_cams:
            rvec, _ = cv2.Rodrigues(Rs_d[c])
            x.extend(rvec.flatten())
            x.extend(Ts_d[c].flatten())
        return np.array(x, dtype=np.float64)

    def _unpack(x):
        Rs_u, Ts_u = {cam_a: Rs[cam_a]}, {cam_a: Ts[cam_a]}
        for i, c in enumerate(free_cams):
            off = i * 6
            rvec = x[off:off+3].reshape(3,1)
            tvec = x[off+3:off+6].reshape(3,1)
            Rs_u[c], _ = cv2.Rodrigues(rvec)
            Ts_u[c]    = tvec
        return Rs_u, Ts_u

    def _residuals(x):
        Rs_u, Ts_u = _unpack(x)
        res = []
        for cam, X3d, p2d in obs:
            if cam not in Rs_u:
                continue
            p = _project(Ks[cam], Rs_u[cam], Ts_u[cam].flatten(), X3d)
            res.extend((p - p2d).tolist())
        return res

    x0    = _pack(Rs, Ts)
    res_before = np.median(np.abs(_residuals(x0)))
    result = least_squares(_residuals, x0, method='trf', loss='soft_l1', max_nfev=2000)
    res_after  = np.median(np.abs(_residuals(result.x)))
    Rs, Ts = _unpack(result.x)
    print(f"  Reprojection error: {res_before:.1f}px → {res_after:.1f}px (median)")

    # ── 5.9  Extract reconstructed camera centres ────────────────────────────
    reconstructed = {cam: _cam_center(Rs[cam], Ts[cam]) for cam in placed}
    print(f"\n  Reconstructed camera centres (self-calib frame):")
    for cam in CAMERA_IDS:
        if cam in reconstructed:
            c = reconstructed[cam]
            print(f"    cam{cam}: [{c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}] mm")

    # ── 5.10 Procrustes alignment → room frame ───────────────────────────────
    common = [c for c in CAMERA_IDS if c in reconstructed]
    src    = np.array([reconstructed[c]        for c in common])
    tgt    = np.array([KNOWN_POSITIONS[c]       for c in common])

    R_align, t_align = _procrustes_rotation(src, tgt)
    aligned = {c: R_align @ reconstructed[c] + t_align for c in common}

    # ── 5.11 Compute errors ──────────────────────────────────────────────────
    errors = {c: float(np.linalg.norm(aligned[c] - KNOWN_POSITIONS[c])) for c in common}
    mean_err = np.mean(list(errors.values()))
    print(f"\n  Position error after Procrustes alignment (mm):")
    for cam in CAMERA_IDS:
        if cam in errors:
            a = aligned[cam]
            k = KNOWN_POSITIONS[cam]
            print(f"    cam{cam}: reconstructed=({a[0]:.0f},{a[1]:.0f},{a[2]:.0f})  "
                  f"known=({k[0]:.0f},{k[1]:.0f},{k[2]:.0f})  "
                  f"error={errors[cam]:.0f} mm")
    print(f"  Mean error: {mean_err:.0f} mm")

    # ── 5.11b  Floor pin: shift world z so ankle keypoints sit on z=0 ────────
    # Triangulate ankles (kp 15,16) across all frames using current world-frame
    # projection matrices, then translate the world so the 5th-percentile foot
    # height becomes 0. This pins the floor without changing camera geometry.
    print("\n  Floor pin (ankles → z=0):")
    P_world = {}
    for c in placed:
        R_w_c = Rs[c] @ R_align.T
        T_w_c = Ts[c].flatten() - Rs[c] @ R_align.T @ t_align
        P_world[c] = Ks[c] @ np.hstack([R_w_c, T_w_c.reshape(3, 1)])

    ankle_zs = []
    all_frames = set()
    for c in placed:
        all_frames.update(idx2det[c].keys())
    for fidx in all_frames:
        for k in (15, 16):
            obs_P, obs_p = [], []
            for c in placed:
                if fidx not in idx2det[c]:
                    continue
                kp = np.array(idx2det[c][fidx][1]).reshape(17, 3)
                if kp[k, 2] >= KP_CONF:
                    obs_P.append(P_world[c])
                    obs_p.append(kp[k, :2])
            if len(obs_P) >= 2:
                rows = []
                for P, p in zip(obs_P, obs_p):
                    rows += [p[0]*P[2] - P[0], p[1]*P[2] - P[1]]
                _, _, Vt = np.linalg.svd(np.stack(rows))
                X = Vt[-1]
                X = X[:3] / X[3]
                ankle_zs.append(X[2])

    if ankle_zs:
        z_floor = float(np.percentile(ankle_zs, 5))
        print(f"    {len(ankle_zs)} ankle observations, "
              f"5th-percentile z={z_floor:.0f} mm  →  shift dz={-z_floor:.0f} mm")
        # Translate world by [0,0,-z_floor]: aligned' = aligned + [0,0,dz],
        # so t_align gains dz in its z component.
        t_align = t_align + np.array([0.0, 0.0, -z_floor])
        # Recompute aligned positions and errors.
        aligned = {c: R_align @ reconstructed[c] + t_align for c in common}
        errors  = {c: float(np.linalg.norm(aligned[c] - KNOWN_POSITIONS[c]))
                   for c in common}
        mean_err = float(np.mean(list(errors.values())))
        print(f"    Mean camera error after floor pin: {mean_err:.0f} mm")
    else:
        print("    No ankle observations available — skipping floor pin")

    # ── 5.12 Save results ────────────────────────────────────────────────────
    result_data = {
        "session":        sess,
        "person_height_mm": PERSON_HEIGHT_MM,
        "anchor_pair":    [cam_a, cam_b],
        "scale_factor":   float(scale),
        "reproj_before_ba": float(res_before),
        "reproj_after_ba":  float(res_after),
        "cameras": {}
    }
    for cam in CAMERA_IDS:
        entry = {
            "known_mm":         KNOWN_POSITIONS[cam].tolist(),
            "placed":           cam in placed,
        }
        if cam in aligned:
            entry["reconstructed_aligned_mm"] = [round(v, 1) for v in aligned[cam].tolist()]
            entry["error_mm"]                 = round(errors[cam], 1)
        # World-frame projection data for triangulation:
        #   X_world → camera:  x_cam = K @ (R_world @ X_world + T_world)
        #   R_world = R_ba @ R_align.T
        #   T_world = T_ba - R_ba @ R_align.T @ t_align
        if cam in placed:
            R_w = Rs[cam] @ R_align.T
            T_w = Ts[cam].flatten() - Rs[cam] @ R_align.T @ t_align
            entry["R_world"] = [[round(v, 8) for v in row] for row in R_w.tolist()]
            entry["T_world"] = [round(v, 4) for v in T_w.tolist()]
            entry["K"]       = [[round(v, 4) for v in row] for row in Ks[cam].tolist()]
        result_data["cameras"][cam] = entry

    result_data["mean_error_mm"] = round(mean_err, 1)

    out_path = OUT / "calibration_result.json"
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  Saved: {out_path}")
    return result_data


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6 — Checkerboard detection (cv2.findChessboardCornersSB)
# ─────────────────────────────────────────────────────────────────────────────

CB_W          = 13    # inner corners along width
CB_H          = 9     # inner corners along height
CB_SQUARE_MM  = 40.0  # physical square size in mm

# 3D object points for one board: (0,0,0), (40,0,0), ..., (480,320,0)
CB_OBJ_PTS = np.zeros((CB_W * CB_H, 3), dtype=np.float32)
CB_OBJ_PTS[:, :2] = np.mgrid[0:CB_W, 0:CB_H].T.reshape(-1, 2) * CB_SQUARE_MM

CB_FLAGS_SB = (cv2.CALIB_CB_NORMALIZE_IMAGE |
               cv2.CALIB_CB_EXHAUSTIVE      |
               cv2.CALIB_CB_ACCURACY)
CB_FLAGS_ORIG = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def _detect_one_camera(args):
    """Worker: scan all frames of one video for checkerboard corners."""
    sess, cam, video_path, out_path, n_total = args
    out_path = Path(out_path)

    # ── resume: load existing file ────────────────────────────────────────────
    if out_path.exists():
        detections = []
        with open(out_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        i = 0
        while i + 1 < len(lines):
            fidx    = int(lines[i].split()[0])
            corners = np.array(list(map(float, lines[i+1].split())),
                               dtype=np.float32).reshape(-1, 2)
            detections.append((fidx, corners))
            i += 2
        print(f"  {sess}/cam{cam}: loaded {len(detections)}/{n_total} from cache",
              flush=True)
        return sess, cam, detections

    # ── detect from video, every frame ───────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    detections = []
    found_total = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    print(f"  {sess}/cam{cam}: scanning {n_total} frames…", flush=True)

    with open(out_path, "w") as f:
        fidx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(gray, (CB_W, CB_H), CB_FLAGS_ORIG)
            if ok:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                found_total += 1
                corners = corners.reshape(-1, 2)
                f.write(f"{fidx} {len(corners)}\n")
                f.write(" ".join(f"{v:.3f}" for v in corners.flatten()) + "\n")
                f.flush()
                detections.append((fidx, corners))
                print(f"  ✓ {sess}/cam{cam}  frame {fidx:5d}/{n_total}"
                      f"  found: {found_total}", flush=True)
            elif fidx % 100 == 0:
                print(f"  · {sess}/cam{cam}  frame {fidx:5d}/{n_total}"
                      f"  found: {found_total}", flush=True)
            fidx += 1

    cap.release()
    pct = 100 * found_total / max(1, n_total)
    print(f"  ✓✓ {sess}/cam{cam}: DONE — {found_total}/{n_total} ({pct:.1f}%)",
          flush=True)
    return sess, cam, detections


def stage6_checkerboard(meta):
    """Scan every frame of every video for checkerboard corners — parallel workers."""
    n_workers = max(1, int(os.cpu_count() * 0.8))
    print(f"\n── Stage 6: Checkerboard — ALL frames, {CB_W}×{CB_H}, "
          f"{n_workers}/{os.cpu_count()} workers ──")

    tasks = [
        (sess, cam, str(SESSIONS[sess][cam]),
         str(OUT / f"checkerboard_{cam}_{sess}.txt"),
         meta[sess][cam]["n_frames"])
        for sess in ["250707", "250708"]
        for cam in CAMERA_IDS
    ]

    cb_flat = {}
    with mp.Pool(processes=n_workers) as pool:
        for sess, cam, detections in pool.imap_unordered(_detect_one_camera, tasks):
            cb_flat[(sess, cam)] = detections

    cb_data = {
        sess: {cam: cb_flat[(sess, cam)] for cam in CAMERA_IDS}
        for sess in ["250707", "250708"]
    }

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n  ── Detection summary ──")
    for sess in ["250707", "250708"]:
        for cam in CAMERA_IDS:
            n   = len(cb_data[sess][cam])
            tot = meta[sess][cam]["n_frames"]
            print(f"    {sess}/cam{cam}: {n}/{tot} ({100*n/max(1,tot):.1f}%)")

    # ── simultaneous detections ───────────────────────────────────────────────
    print(f"\n  ── Simultaneous detections (cross-camera, same frame) ──")
    for sess in ["250707", "250708"]:
        sets = {cam: {fidx for fidx, _ in cb_data[sess][cam]} for cam in CAMERA_IDS}
        for i, ca in enumerate(CAMERA_IDS):
            for cb_cam in CAMERA_IDS[i+1:]:
                n = len(sets[ca] & sets[cb_cam])
                print(f"    {sess}: cam{ca}+cam{cb_cam}: {n} simultaneous frames")

    return cb_data


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7 — Checkerboard self-calibration
# ─────────────────────────────────────────────────────────────────────────────
def stage7_cb_calibrate(cb_data):
    print("\n── Stage 7: Checkerboard self-calibration ──")

    # ── Per-camera intrinsic calibration ─────────────────────────────────────
    intrinsics = {}
    for cam in CAMERA_IDS:
        obj_pts, img_pts = [], []
        for sess in ["250707", "250708"]:
            for fidx, corners in cb_data[sess][cam]:
                obj_pts.append(CB_OBJ_PTS)
                img_pts.append(corners.reshape(-1, 1, 2))
        n = len(obj_pts)
        if n < 5:
            print(f"  cam{cam}: only {n} boards — skipping")
            continue
        ret, K, dist, _, _ = cv2.calibrateCamera(
            obj_pts, img_pts, (2048, 1496), None, None)
        intrinsics[cam] = {"K": K, "dist": dist, "reproj_px": round(ret, 2), "n_boards": n}
        print(f"  cam{cam} ({n} boards):  reproj={ret:.2f}px  "
              f"fx={K[0,0]:.0f}  fy={K[1,1]:.0f}  cx={K[0,2]:.0f}  cy={K[1,2]:.0f}")

    # ── Stereo calibration: best simultaneous pair ────────────────────────────
    # Find pair with most simultaneous frames across both sessions
    best_n, best_pair, best_sess = 0, None, None
    for sess in ["250707", "250708"]:
        sets = {cam: {fidx for fidx, _ in cb_data[sess][cam]} for cam in CAMERA_IDS}
        for i, ca in enumerate(CAMERA_IDS):
            for cb_cam in CAMERA_IDS[i+1:]:
                n = len(sets[ca] & sets[cb_cam])
                if n > best_n and ca in intrinsics and cb_cam in intrinsics:
                    best_n, best_pair, best_sess = n, (ca, cb_cam), sess

    if best_pair is None or best_n < 5:
        print("  Not enough simultaneous frames for stereo calibration")
        return None

    cam_a, cam_b = best_pair
    sess = best_sess
    print(f"\n  Stereo: cam{cam_a}+cam{cam_b}  session={sess}  frames={best_n}")

    frames_a = {fidx: corners for fidx, corners in cb_data[sess][cam_a]}
    frames_b = {fidx: corners for fidx, corners in cb_data[sess][cam_b]}
    common   = sorted(set(frames_a) & set(frames_b))

    K_a, D_a = intrinsics[cam_a]["K"], intrinsics[cam_a]["dist"]
    K_b, D_b = intrinsics[cam_b]["K"], intrinsics[cam_b]["dist"]

    obj_pts = [CB_OBJ_PTS              for f in common]
    img_a   = [frames_a[f].reshape(-1,1,2) for f in common]
    img_b   = [frames_b[f].reshape(-1,1,2) for f in common]

    ret, K_a, D_a, K_b, D_b, R_rel, T_rel, E, F = cv2.stereoCalibrate(
        obj_pts, img_a, img_b, K_a, D_a, K_b, D_b, (2048, 1496),
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-7))

    T_rel = T_rel.flatten()
    dist_measured = float(np.linalg.norm(T_rel))

    P_a = KNOWN_POSITIONS[cam_a]
    P_b = KNOWN_POSITIONS[cam_b]
    dist_gt  = float(np.linalg.norm(P_b - P_a))
    dist_err = dist_measured - dist_gt

    print(f"  Stereo reproj:  {ret:.2f} px")
    print(f"  Distance cam{cam_a}↔cam{cam_b}:  measured={dist_measured:.0f} mm  "
          f"GT={dist_gt:.0f} mm  err={dist_err:+.0f} mm ({100*dist_err/dist_gt:+.1f}%)")

    # ── Reconstruct positions in world frame ──────────────────────────────────
    # Anchor cam_a at known position.
    # T_rel is the vector to cam_b in cam_a's camera frame.
    # Use Procrustes (rotation only) to align the 2-point cloud to world frame.
    # Source: [{0,0,0}, T_rel] in cam_a frame
    # Target: [P_a, P_b] in world frame — centroid-subtracted for rotation recovery
    src = np.stack([np.zeros(3), T_rel])
    tgt = np.stack([P_a, P_b])
    src_c = src - src.mean(0)
    tgt_c = tgt - tgt.mean(0)
    U, _, Vt = np.linalg.svd(tgt_c.T @ src_c)
    R_world = U @ Vt
    if np.linalg.det(R_world) < 0:
        U[:, -1] *= -1
        R_world = U @ Vt

    src_world  = (R_world @ src.T).T + (P_a - (R_world @ np.zeros(3)))
    # cam_a is placed at P_a, cam_b at:
    cam_b_recon = src_world[1]

    err_a = float(np.linalg.norm(src_world[0] - P_a))
    err_b = float(np.linalg.norm(cam_b_recon  - P_b))
    print(f"  cam{cam_a} recon: {src_world[0].round(0).tolist()}  err={err_a:.0f} mm  (anchor)")
    print(f"  cam{cam_b} recon: {cam_b_recon.round(0).tolist()}   err={err_b:.0f} mm")
    print(f"  cam108, cam117: no simultaneous data — not calibrated")

    # ── Save result ───────────────────────────────────────────────────────────
    result = {
        "session": sess,
        "method": "checkerboard",
        "stereo_pair": [cam_a, cam_b],
        "n_simultaneous_frames": len(common),
        "stereo_reproj_px": round(ret, 2),
        "distance_measured_mm": round(dist_measured, 1),
        "distance_gt_mm": round(dist_gt, 1),
        "distance_error_mm": round(abs(dist_err), 1),
        "intrinsics": {
            cam: {
                "fx": round(float(v["K"][0, 0]), 1),
                "fy": round(float(v["K"][1, 1]), 1),
                "cx": round(float(v["K"][0, 2]), 1),
                "cy": round(float(v["K"][1, 2]), 1),
                "reproj_px": v["reproj_px"],
                "n_boards": v["n_boards"],
            }
            for cam, v in intrinsics.items()
        },
        "cameras": {
            cam_a: {
                "placed": True,
                "reconstructed_aligned_mm": [round(float(x), 1) for x in src_world[0]],
                "known_mm": [round(float(x), 1) for x in P_a],
                "error_mm": round(err_a, 1),
            },
            cam_b: {
                "placed": True,
                "reconstructed_aligned_mm": [round(float(x), 1) for x in cam_b_recon],
                "known_mm": [round(float(x), 1) for x in P_b],
                "error_mm": round(err_b, 1),
            },
            **{cam: {"placed": False} for cam in CAMERA_IDS
               if cam not in (cam_a, cam_b)},
        },
    }

    out = OUT / "checkerboard_result.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    meta      = stage1_inspect()
    frame_map = stage2_extract(meta)
    pose_data = stage3_detect(frame_map)
    preview   = stage4_preview(frame_map, pose_data)
    calib     = stage5_calibrate(pose_data, meta)
    cb_data   = stage6_checkerboard(meta)
    cb_calib  = stage7_cb_calibrate(cb_data)
    print(f"\nDone.  Open preview:\n  open '{preview}'")
