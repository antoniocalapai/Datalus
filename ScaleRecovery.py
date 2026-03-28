#!/usr/bin/env python3
"""
ScaleRecovery.py — Recover real-world scale from person height (1700 mm)

Stages:
  1  Load        — existing PriCaB poses, intrinsics, detections
  2  ValidFrames — full-body frames visible in ≥2 cameras
  3  Triangulate — head-to-ankle distance in arbitrary units
  4  Scale       — compute scale factor (1700 / median_height)
  5  Apply       — scale T vectors, save pricab_poses_scaled.npz
  6  Validate    — inter-camera distances, heights, reprojection
  7  YAMLs       — overwrite PriCaB yamls with scaled poses
  8  Viewer      — rebuild pricab_viewer.html with trajectory + overlay
  9  Config      — update datalus_config.json

Usage:
    python3 ScaleRecovery.py
"""

import base64
import json
import random
import statistics
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
PRICAB_OUT     = BASE_DIR / "PriCaB_output"
INTRINSICS_NPZ = BASE_DIR / "DatalusCalibration" / "intrinsics.npz"
CONFIG_JSON    = BASE_DIR / "datalus_config.json"

POSES_NPZ    = PRICAB_OUT / "pricab_poses.npz"
POSES_SCALED = PRICAB_OUT / "pricab_poses_scaled.npz"
SCALE_REPORT = PRICAB_OUT / "scale_report.txt"
VIEWER_HTML  = PRICAB_OUT / "pricab_viewer.html"
YAML_DIR     = PRICAB_OUT / "yamls"

# ─── Constants ─────────────────────────────────────────────────────────────────
PERSON_HEIGHT_MM   = 1700.0
LENS_FOCAL_MM      = 8.0
SENSOR_WIDTH_MM    = 11.2
IMG_WIDTH          = 2048
IMG_HEIGHT         = 1500
IMG_SCALE          = 0.25
JPEG_QUALITY       = 40

# Keypoints used for scale: nose(0), lsh(5), rsh(6), lhip(11), rhip(12), lank(15), rank(16)
SCALE_KPS          = [0, 5, 6, 11, 12, 15, 16]
CONF_THRESH_HIGH   = 0.6
CONF_THRESH_LOW    = 0.5
BBOX_CONF_THRESH   = 0.7
VERTICAL_SPAN_FRAC = 0.30
MIN_VALID_FRAMES   = 20
TRAJ_LEN           = 30


def _header(n, title):
    print(f"\n{'═'*60}")
    print(f"STAGE {n} — {title}")
    print('═'*60)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Load existing reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def stage1_load():
    _header(1, "LOAD EXISTING RECONSTRUCTION")

    if not POSES_NPZ.exists():
        sys.exit(f"[ERROR] {POSES_NPZ} not found. Run PriCaB_HumanCalib.py first.")

    npz     = np.load(str(POSES_NPZ))
    ref_cam = str(npz["reference_camera"][0])
    reproj  = float(npz["reprojection_error_px"][0])
    cam_ids = sorted(k[:-2] for k in npz.files if k.endswith("_R"))

    print(f"  Cameras ({len(cam_ids)}): {cam_ids}")
    print(f"  Reference: {ref_cam}   original reproj error: {reproj:.1f} px")

    # Load intrinsics
    loaded_K, loaded_dist = {}, {}
    if INTRINSICS_NPZ.exists():
        inpz        = np.load(str(INTRINSICS_NPZ))
        loaded_K    = {k[:-2]: inpz[k] for k in inpz.files if k.endswith("_K")}
        loaded_dist = {k[:-5]: inpz[k] for k in inpz.files if k.endswith("_dist")}

    cams = {}
    for cam_id in cam_ids:
        R = npz[f"{cam_id}_R"].astype(np.float64)
        T = npz[f"{cam_id}_T"].reshape(3, 1).astype(np.float64)

        if cam_id in loaded_K:
            K    = loaded_K[cam_id].astype(np.float64)
            dist = loaded_dist.get(cam_id, np.zeros((1, 5))).ravel().astype(np.float64)
            src  = "npz"
        else:
            fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) * IMG_WIDTH
            K  = np.array([[fx, 0, IMG_WIDTH / 2.0],
                           [0, fx, IMG_HEIGHT / 2.0],
                           [0,  0, 1.0]], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            src  = "estimated"

        P = K @ np.hstack([R, T])
        C = (-R.T @ T).ravel()
        cams[cam_id] = {"K": K, "dist": dist, "R": R, "T": T, "P": P, "C": C, "src": src}

    # Load detections
    detections = {}
    for cam_id in cam_ids:
        txt      = PRICAB_OUT / f"pose_{cam_id}.txt"
        cam_dets = {}
        if txt.exists():
            for line in txt.read_text().splitlines():
                cols = line.strip().split()
                if len(cols) < 57:
                    continue
                fidx      = int(cols[0])
                bbox      = [float(c) for c in cols[1:5]]
                bbox_conf = float(cols[5])
                kps       = np.array(cols[6:57], dtype=np.float32).reshape(17, 3)
                cam_dets[fidx] = {"bbox": bbox, "bbox_conf": bbox_conf, "kps": kps}
        detections[cam_id] = cam_dets
        print(f"  cam {cam_id}: {len(cam_dets):3d} frames  [{cams[cam_id]['src']}]")

    return cam_ids, cams, detections, ref_cam, reproj


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Find valid full-body frames
# ═══════════════════════════════════════════════════════════════════════════════

def _det_is_valid(det, kp_conf_thresh, bbox_conf_thresh):
    if det["bbox_conf"] < bbox_conf_thresh:
        return False
    kps = det["kps"]
    for ki in SCALE_KPS:
        if kps[ki, 2] < kp_conf_thresh:
            return False
    y_vals = [kps[ki, 1] for ki in SCALE_KPS]
    if (max(y_vals) - min(y_vals)) < VERTICAL_SPAN_FRAC * IMG_HEIGHT:
        return False
    return True


def stage2_find_valid_frames(cam_ids, detections):
    _header(2, "FIND VALID FULL-BODY FRAMES")

    result_frames = None
    for kp_thresh in [CONF_THRESH_HIGH, CONF_THRESH_LOW]:
        valid = {}   # frame -> [cam_id, ...]
        for cam_id in cam_ids:
            for fidx, det in detections[cam_id].items():
                if _det_is_valid(det, kp_thresh, BBOX_CONF_THRESH):
                    valid.setdefault(fidx, []).append(cam_id)

        multi = {f: cs for f, cs in valid.items() if len(cs) >= 2}
        print(f"  kp_conf={kp_thresh}: {len(multi)} frames with ≥2 valid cameras")

        if len(multi) >= MIN_VALID_FRAMES:
            result_frames = multi
            break

    if result_frames is None or len(result_frames) < MIN_VALID_FRAMES:
        n = len(result_frames) if result_frames else 0
        cam_det_counts = {c: len(detections[c]) for c in cam_ids}
        sys.exit(
            f"[ERROR] Only {n} valid frames found (threshold tried: 0.6, 0.5).\n"
            f"Detection counts per camera: {cam_det_counts}\n"
            f"Increase video coverage or lower VERTICAL_SPAN_FRAC."
        )

    # Report top camera pairs
    pair_counts = {}
    for vcams in result_frames.values():
        for ca, cb in combinations(sorted(vcams), 2):
            pair_counts[(ca, cb)] = pair_counts.get((ca, cb), 0) + 1
    print(f"  Top camera pairs by valid frame count:")
    for (ca, cb), cnt in sorted(pair_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {ca}↔{cb}: {cnt}")

    return result_frames, kp_thresh


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Triangulate full-body height
# ═══════════════════════════════════════════════════════════════════════════════

def _tri_kp(Pa, Pb, xa, ya, xb, yb):
    X4 = cv2.triangulatePoints(
        Pa, Pb,
        np.array([[xa], [ya]], dtype=np.float64),
        np.array([[xb], [yb]], dtype=np.float64),
    )
    w = float(X4[3])
    if abs(w) < 1e-10:
        return None
    X3 = (X4[:3] / w).ravel()
    return X3 if np.all(np.isfinite(X3)) else None


def stage3_triangulate(cam_ids, cams, detections, valid_frames, kp_thresh):
    _header(3, "TRIANGULATE FULL-BODY HEIGHT PER FRAME")

    head_ankle, shoulder_w, hip_w = [], [], []
    per_frame_heights = {}   # frame -> median head-ankle in raw units

    for fidx, vcams in sorted(valid_frames.items()):
        frame_ha = []
        for ca, cb in combinations(sorted(vcams), 2):
            det_a = detections[ca][fidx]
            det_b = detections[cb][fidx]
            Pa    = cams[ca]["P"]
            Pb    = cams[cb]["P"]
            Ra, Ta = cams[ca]["R"], cams[ca]["T"].ravel()
            Rb, Tb = cams[cb]["R"], cams[cb]["T"].ravel()

            pts3d = {}
            ok = True
            for ki in SCALE_KPS:
                xa, ya = float(det_a["kps"][ki, 0]), float(det_a["kps"][ki, 1])
                xb, yb = float(det_b["kps"][ki, 0]), float(det_b["kps"][ki, 1])
                X3 = _tri_kp(Pa, Pb, xa, ya, xb, yb)
                if X3 is None:
                    ok = False; break
                # Cheirality
                if (Ra @ X3 + Ta)[2] <= 0 or (Rb @ X3 + Tb)[2] <= 0:
                    ok = False; break
                pts3d[ki] = X3
            if not ok:
                continue

            ha = float(np.linalg.norm(pts3d[0] - (pts3d[15] + pts3d[16]) / 2))
            sw = float(np.linalg.norm(pts3d[5] - pts3d[6]))
            hw = float(np.linalg.norm(pts3d[11] - pts3d[12]))

            head_ankle.append(ha)
            shoulder_w.append(sw)
            hip_w.append(hw)
            frame_ha.append(ha)

        if frame_ha:
            per_frame_heights[fidx] = float(statistics.median(frame_ha))

    print(f"  Measurements: {len(head_ankle)} samples from {len(per_frame_heights)} frames")
    if not head_ankle:
        sys.exit("[ERROR] Zero valid triangulations. Check camera poses and keypoint quality.")

    return head_ankle, shoulder_w, hip_w, per_frame_heights


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 4 — Compute scale factor
# ═══════════════════════════════════════════════════════════════════════════════

def stage4_compute_scale(head_ankle, shoulder_w):
    _header(4, "COMPUTE ROBUST SCALE FACTOR")

    arr        = np.array(head_ankle)
    mean_val   = float(np.mean(arr))
    median_val = float(np.median(arr))
    std_val    = float(np.std(arr))
    p10        = float(np.percentile(arr, 10))
    p90        = float(np.percentile(arr, 90))
    n          = len(arr)

    print(f"  Head-to-ankle distance (arbitrary units):")
    print(f"    N        : {n}")
    print(f"    Mean     : {mean_val:.5f}")
    print(f"    Median   : {median_val:.5f}  ← used for scale")
    print(f"    Std      : {std_val:.5f}")
    print(f"    10th pct : {p10:.5f}")
    print(f"    90th pct : {p90:.5f}")

    scale = PERSON_HEIGHT_MM / median_val
    print(f"\n  Scale factor : {scale:.4f}")
    print(f"  1 unit       = {1000.0 / scale:.2f} mm")

    if shoulder_w:
        sw_mm = float(np.median(shoulder_w)) * scale
        ok = 350 <= sw_mm <= 500
        mark = "✓" if ok else "⚠ WARNING"
        print(f"  Shoulder width validation: {sw_mm:.0f} mm  {mark} (expected 350–500 mm)")

    return scale, median_val, n


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 5 — Apply scale, save
# ═══════════════════════════════════════════════════════════════════════════════

def stage5_apply_scale(cam_ids, cams, scale, median_ha, n_samples,
                        head_ankle, shoulder_w, hip_w):
    _header(5, "APPLY SCALE FACTOR TO TRANSLATIONS")

    cams_s = {}
    for cam_id in cam_ids:
        R = cams[cam_id]["R"].copy()
        T = cams[cam_id]["T"].copy() * scale
        K = cams[cam_id]["K"].copy()
        C = (-R.T @ T).ravel()
        P = K @ np.hstack([R, T])
        cams_s[cam_id] = {**cams[cam_id], "T": T, "C": C, "P": P}

    # Save npz (preserve reference_camera from original)
    orig = np.load(str(POSES_NPZ))
    save = {
        "reference_camera":      orig["reference_camera"],
        "reprojection_error_px": orig["reprojection_error_px"],
        "scale_factor":          np.array([scale]),
    }
    for cam_id in cam_ids:
        save[f"{cam_id}_R"] = cams_s[cam_id]["R"]
        save[f"{cam_id}_T"] = cams_s[cam_id]["T"]
    np.savez(str(POSES_SCALED), **save)
    print(f"  Saved → {POSES_SCALED}")

    # Text report
    sw_mm  = float(np.median(shoulder_w)) * scale if shoulder_w else 0.0
    hw_mm  = float(np.median(hip_w))      * scale if hip_w      else 0.0
    report = [
        "PriCaB Scale Recovery Report",
        "=" * 44,
        f"Scale factor                   : {scale:.6f}",
        f"Person height assumed (mm)      : {PERSON_HEIGHT_MM:.0f}",
        f"Samples used                    : {n_samples}",
        f"Median head-ankle (raw units)   : {median_ha:.6f}",
        f"Median head-ankle (scaled mm)   : {median_ha * scale:.1f}",
        f"Mean head-ankle  (raw units)    : {float(np.mean(head_ankle)):.6f}",
        f"Std  head-ankle  (raw units)    : {float(np.std(head_ankle)):.6f}",
        f"Shoulder width (scaled mm)      : {sw_mm:.1f}",
        f"Hip width      (scaled mm)      : {hw_mm:.1f}",
        "",
        f"{'Camera':<8} {'|T| raw':>10} {'|T| scaled mm':>15}",
        "-" * 36,
    ]
    for cam_id in sorted(cam_ids):
        t_raw = float(np.linalg.norm(cams[cam_id]["T"]))
        t_mm  = float(np.linalg.norm(cams_s[cam_id]["T"]))
        report.append(f"  {cam_id:<6} {t_raw:>10.4f} {t_mm:>15.1f}")
        print(f"  cam {cam_id}: |T| {t_raw:.4f} → {t_mm:.1f} mm")

    SCALE_REPORT.write_text("\n".join(report) + "\n")
    print(f"  Report → {SCALE_REPORT}")
    return cams_s


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 6 — Validate scaled reconstruction
# ═══════════════════════════════════════════════════════════════════════════════

def stage6_validate(cam_ids, cams_s, detections, valid_frames):
    _header(6, "VALIDATE SCALED RECONSTRUCTION")

    centres = {c: cams_s[c]["C"] for c in cam_ids}

    # Check 1 — Inter-camera distances
    print("\n  CHECK 1 — Inter-camera distances (mm):")
    cam_ids_s = sorted(cam_ids)
    for ca, cb in combinations(cam_ids_s, 2):
        d    = float(np.linalg.norm(centres[ca] - centres[cb]))
        flag = ("  ⚠ TOO CLOSE" if d < 200 else
                "  ⚠ TOO FAR"  if d > 6000 else "")
        print(f"    {ca}↔{cb}: {d:7.0f} mm{flag}")

    # Check 2 — Camera height (Z after Rx(-90°): z_up = -old_y)
    print("\n  CHECK 2 — Camera Z height (mm, Z=up after rotation):")
    z_vals = {}
    for cam_id in cam_ids_s:
        z_up = float(-centres[cam_id][1])   # Rx(-90°): new_z = -old_y
        z_vals[cam_id] = z_up
        print(f"    cam {cam_id}: Z = {z_up:8.1f} mm")
    z_arr = np.array(list(z_vals.values()))
    z_med = float(np.median(z_arr))
    z_mad = float(np.median(np.abs(z_arr - z_med))) + 1e-3
    for cam_id, z in z_vals.items():
        if abs(z - z_med) > 3 * z_mad + 200:
            print(f"    ⚠ cam {cam_id}: Z={z:.0f} outlier (median={z_med:.0f})")

    # Check 3 — Reprojection error at scale
    print("\n  CHECK 3 — Reprojection error (10 random valid frames):")
    random.seed(42)
    test_frames = random.sample(sorted(valid_frames), min(10, len(valid_frames)))
    all_errs = []
    for fidx in sorted(test_frames):
        vcams = valid_frames[fidx]
        if len(vcams) < 2:
            continue
        ca, cb = sorted(vcams)[:2]
        Pa, Pb = cams_s[ca]["P"], cams_s[cb]["P"]
        Ra, Ta = cams_s[ca]["R"], cams_s[ca]["T"].ravel()

        pts = []
        for ki in [11, 12]:
            xa = float(detections[ca][fidx]["kps"][ki, 0])
            ya = float(detections[ca][fidx]["kps"][ki, 1])
            xb = float(detections[cb][fidx]["kps"][ki, 0])
            yb = float(detections[cb][fidx]["kps"][ki, 1])
            X3 = _tri_kp(Pa, Pb, xa, ya, xb, yb)
            if X3 is not None and (Ra @ X3 + Ta)[2] > 0:
                pts.append(X3)
        if not pts:
            continue
        X3 = np.mean(pts, axis=0)

        for cam_id in vcams:
            if fidx not in detections[cam_id]:
                continue
            R, T, K = cams_s[cam_id]["R"], cams_s[cam_id]["T"].ravel(), cams_s[cam_id]["K"]
            rvec, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(
                X3.reshape(1, 1, 3), rvec, T, K, cams_s[cam_id]["dist"])
            px, py = proj.ravel()
            for ki in [11, 12]:
                gx, gy, gc = detections[cam_id][fidx]["kps"][ki]
                if gc > 0.3:
                    all_errs.append(float(np.hypot(px - gx, py - gy)))

    if all_errs:
        mean_e = float(np.mean(all_errs))
        max_e  = float(np.max(all_errs))
        print(f"    Mean: {mean_e:.1f} px   Max: {max_e:.1f} px")
    else:
        mean_e = 0.0
        print("    No reprojection data.")

    return mean_e


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7 — Rewrite YAMLs with scaled poses
# ═══════════════════════════════════════════════════════════════════════════════

def stage7_write_yamls(cam_ids, cams_s):
    _header(7, "REWRITE YAMLs WITH SCALED POSES")
    YAML_DIR.mkdir(parents=True, exist_ok=True)

    for cam_id in sorted(cam_ids):
        out  = YAML_DIR / f"{cam_id}.yaml"
        K    = cams_s[cam_id]["K"]
        dist = cams_s[cam_id]["dist"]
        R    = cams_s[cam_id]["R"]
        T    = cams_s[cam_id]["T"]
        # ABT reads K.mat().T and R.mat().T → store transposes
        fs = cv2.FileStorage(str(out), cv2.FILE_STORAGE_WRITE)
        fs.write("intrinsicMatrix",        K.T)
        fs.write("distortionCoefficients", dist.reshape(1, -1))
        fs.write("R",                      R.T)
        fs.write("T",                      T.reshape(3, 1))
        fs.release()
        print(f"  cam {cam_id}: |T|={np.linalg.norm(T):.1f} mm → {out.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Rebuild viewer
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_img(path, sc, q):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    h, w  = img.shape[:2]
    small = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _zup(v):
    """Rx(-90°): (x,y,z) → (x, z, -y). Converts to Z-up world frame."""
    return [float(v[0]), float(v[2]), float(-v[1])]


def _room_wireframe(centres_zup):
    xs = [c[0] for c in centres_zup]
    ys = [c[1] for c in centres_zup]
    zs = [c[2] for c in centres_zup]

    def expand(lo, hi, frac=0.20):
        span = max(hi - lo, 500.0)
        return lo - span * frac, hi + span * frac

    x0, x1 = expand(min(xs), max(xs))
    y0, y1 = expand(min(ys), max(ys))
    z0      = 0.0
    z1      = max(zs) * 1.25 if max(zs) > 0 else abs(min(zs)) * 1.25

    corners = [
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],
    ]
    edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    ex, ey, ez = [], [], []
    for a, b in edges:
        ex += [corners[a][0], corners[b][0], None]
        ey += [corners[a][1], corners[b][1], None]
        ez += [corners[a][2], corners[b][2], None]
    return {"type": "scatter3d", "x": ex, "y": ey, "z": ez,
            "mode": "lines", "name": "Room",
            "line": {"color": "#445566", "width": 2}, "hoverinfo": "skip"}


def stage8_build_viewer(cam_ids, cams_s, detections, scale, per_frame_heights):
    _header(8, "BUILD VIEWER (SCALED RECONSTRUCTION)")

    cam_ids_s   = sorted(cam_ids)
    frames_dirs = {c: PRICAB_OUT / "frames" / c for c in cam_ids_s}

    all_frames    = set()
    for c in cam_ids_s:
        all_frames |= set(detections[c])
    shared_frames = sorted(all_frames)
    print(f"  {len(shared_frames)} frames, {len(cam_ids_s)} cameras")

    # Triangulate hip midpoints
    print("  Triangulating hip midpoints (scaled)…")
    positions3d = {}
    for frame in shared_frames:
        kpc = {c: detections[c][frame]["kps"]
               for c in cam_ids_s if frame in detections[c]}
        if len(kpc) < 2:
            continue
        pts = []
        clist = list(kpc.keys())
        for ca, cb in combinations(clist[:6], 2):
            Ra, Ta = cams_s[ca]["R"], cams_s[ca]["T"].ravel()
            Rb, Tb = cams_s[cb]["R"], cams_s[cb]["T"].ravel()
            for ki in [11, 12]:
                if kpc[ca][ki, 2] < 0.3 or kpc[cb][ki, 2] < 0.3:
                    continue
                X3 = _tri_kp(cams_s[ca]["P"], cams_s[cb]["P"],
                              float(kpc[ca][ki, 0]), float(kpc[ca][ki, 1]),
                              float(kpc[cb][ki, 0]), float(kpc[cb][ki, 1]))
                if X3 is not None and (Ra @ X3 + Ta)[2] > 0 and (Rb @ X3 + Tb)[2] > 0:
                    pts.append(X3)
        if pts:
            med = np.median(pts, axis=0)
            if np.linalg.norm(med) < 1e7:
                positions3d[frame] = _zup(med.tolist())

    print(f"  Hip triangulated: {len(positions3d)}/{len(shared_frames)} frames")

    # Encode images
    print(f"  Encoding {len(shared_frames) * len(cam_ids_s)} images…")
    images = {c: [] for c in cam_ids_s}
    for frame in shared_frames:
        for c in cam_ids_s:
            images[c].append(
                _encode_img(frames_dirs[c] / f"{c}_frame_{frame:06d}.png",
                            IMG_SCALE, JPEG_QUALITY))
    print("  Encoding done.")

    # Bounding boxes for JS
    det_js = {}
    for frame in shared_frames:
        fd = {}
        for c in cam_ids_s:
            if frame in detections[c]:
                fd[c] = detections[c][frame]["bbox"]
        det_js[frame] = fd

    # Camera positions (Z-up)
    cc_zup = {c: _zup(cams_s[c]["C"].tolist()) for c in cam_ids_s}
    centres_list = list(cc_zup.values())

    # Per-frame person height in mm (for overlay)
    pfh_mm = {str(f): round(h * scale, 1) for f, h in per_frame_heights.items()}

    # Traces: 0=room, 1=cameras, 2=hip, 3=trajectory
    first_pos = next((positions3d[f] for f in shared_frames if f in positions3d),
                     centres_list[0])

    traces = [
        _room_wireframe(centres_list),
        {
            "type": "scatter3d",
            "x": [cc_zup[c][0] for c in cam_ids_s],
            "y": [cc_zup[c][1] for c in cam_ids_s],
            "z": [cc_zup[c][2] for c in cam_ids_s],
            "mode": "markers+text", "name": "Cameras",
            "text": [f"cam{c}" for c in cam_ids_s],
            "textposition": "top center",
            "textfont": {"color": "white", "size": 10},
            "marker": {"color": "red", "size": 7, "symbol": "diamond",
                       "line": {"color": "white", "width": 1}},
        },
        {
            "type": "scatter3d",
            "x": [first_pos[0]], "y": [first_pos[1]], "z": [first_pos[2]],
            "mode": "markers", "name": "Hip midpoint",
            "marker": {"color": "#44aaff", "size": 14, "symbol": "circle",
                       "line": {"color": "white", "width": 2}},
        },
        {
            "type": "scatter3d", "x": [], "y": [], "z": [],
            "mode": "lines+markers", "name": "Trajectory",
            "line": {"color": "#1155aa", "width": 3},
            "marker": {"color": [], "size": 4,
                       "colorscale": [[0, "#0a1a3a"], [1, "#44aaff"]],
                       "cmin": 0, "cmax": 1},
            "hoverinfo": "skip",
        },
    ]

    j_layout = json.dumps({
        "paper_bgcolor": "#0a0a1a",
        "font": {"color": "white", "family": "monospace"},
        "scene": {
            "bgcolor": "#0d0d22",
            "xaxis": {"title": "X (floor)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "yaxis": {"title": "Y (floor)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "zaxis": {"title": "Z (up)", "color": "white",
                      "backgroundcolor": "#12122a", "gridcolor": "#2a2a5a"},
            "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2},
                       "up": {"x": 0, "y": 0, "z": 1}},
            "aspectmode": "data",
        },
        "margin": {"l": 0, "r": 0, "b": 0, "t": 30},
        "legend": {"bgcolor": "rgba(20,20,50,0.85)", "font": {"color": "white"}},
        "uirevision": "keep",
    })

    n_cams    = len(cam_ids_s)
    grid_cols = min(n_cams, 4)
    grid_rows = (n_cams + grid_cols - 1) // grid_cols

    cam_divs = "\n    ".join(
        f'<div class="cam-wrap"><canvas id="c{c}"></canvas>'
        f'<div class="cam-label">cam {c}</div></div>'
        for c in cam_ids_s)
    canvas_js = "{\n" + ",\n".join(
        f"  '{c}': document.getElementById('c{c}')" for c in cam_ids_s) + "\n}"

    j_frames     = json.dumps(shared_frames)
    j_images     = json.dumps(images)
    j_detections = json.dumps(det_js)
    j_positions  = json.dumps({str(k): v for k, v in positions3d.items()})
    j_traces     = json.dumps(traces)
    j_cam_ids    = json.dumps(cam_ids_s)
    j_img_dims   = json.dumps({"native_w": IMG_WIDTH, "native_h": IMG_HEIGHT,
                                "disp_w": int(IMG_WIDTH * IMG_SCALE),
                                "disp_h": int(IMG_HEIGHT * IMG_SCALE)})
    j_pfh        = json.dumps(pfh_mm)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PriCaB — Scaled Reconstruction Viewer</title>
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
#main {{ display: flex; flex: 1; overflow: hidden; }}
#cam-panel {{
  width: 50%; display: grid;
  grid-template-columns: repeat({grid_cols}, 1fr);
  grid-template-rows: repeat({grid_rows}, 1fr);
  gap: 2px; background: #050510; padding: 2px;
}}
.cam-wrap {{ position: relative; background: #080818; overflow: hidden; }}
.cam-wrap canvas {{ width: 100%; height: 100%; object-fit: contain; display: block; }}
.cam-label {{
  position: absolute; top: 6px; left: 8px;
  background: rgba(0,0,0,0.65); color: #aabbdd;
  font-size: 11px; padding: 2px 6px; border-radius: 3px; pointer-events: none;
}}
#plot-panel {{ flex: 1; position: relative; }}
#plot3d {{ width: 100%; height: 100%; }}
#overlay {{
  position: absolute; top: 12px; right: 12px;
  background: rgba(8,8,28,0.88); border: 1px solid #334466;
  padding: 9px 13px; border-radius: 6px; font-size: 12px;
  color: #aabbdd; line-height: 1.8; pointer-events: none; min-width: 210px;
}}
#overlay .lbl {{ color: #5577aa; }}
#controls {{
  flex-shrink: 0; background: #0d0d2e;
  border-top: 1px solid #223; padding: 10px 16px;
}}
#ctrl-row {{ display: flex; align-items: center; gap: 12px; }}
.btn {{
  background: #1a2a4a; border: 1px solid #335; color: #aaccff;
  padding: 5px 14px; border-radius: 4px; cursor: pointer;
  font-size: 13px; font-family: monospace;
}}
.btn:hover {{ background: #243560; }}
#slider {{ flex: 1; accent-color: #4477cc; height: 4px; cursor: pointer; }}
#frame-label {{ min-width: 140px; text-align: right; font-size: 12px; color: #7799bb; }}
</style>
</head>
<body>
<div id="header">
  PriCaB — Scaled Viewer &nbsp;|&nbsp; scale={scale:.4f} &nbsp;|&nbsp;
  <span style="color:#ff4444">&#9670; Cameras ({n_cams})</span> &nbsp;
  <span style="color:#44aaff">&#11044; Hip</span>
</div>
<div id="main">
  <div id="cam-panel">
    {cam_divs}
  </div>
  <div id="plot-panel">
    <div id="plot3d"></div>
    <div id="overlay">
      <div><span class="lbl">Frame  </span> <span id="ov-frame">—</span></div>
      <div><span class="lbl">Scale  </span> {scale:.4f}</div>
      <div><span class="lbl">Height </span> <span id="ov-height">—</span> mm</div>
    </div>
  </div>
</div>
<div id="controls">
  <div id="ctrl-row">
    <button class="btn" id="btnPlay">&#9654; Play</button>
    <button class="btn" id="btnPause">&#9646;&#9646; Pause</button>
    <input type="range" id="slider" min="0" max="{len(shared_frames)-1}" value="0">
    <span id="frame-label">frame {shared_frames[0] if shared_frames else 0}</span>
  </div>
</div>

<script>
const FRAMES     = {j_frames};
const IMAGES     = {j_images};
const DETECTIONS = {j_detections};
const POSITIONS  = {j_positions};
const IMG_DIMS   = {j_img_dims};
const CAM_IDS    = {j_cam_ids};
const HEIGHTS    = {j_pfh};
const TRAJ_LEN   = {TRAJ_LEN};

Plotly.newPlot('plot3d', {j_traces}, {j_layout}, {{
  scrollZoom: true, displayModeBar: true,
  modeBarButtonsToRemove: ['toImage'], displaylogo: false,
}});

const canvases = {canvas_js};
const imgCache = {{}};
let posHistory = [];

function getImg(camId, idx) {{
  const key = camId + '_' + idx;
  if (!imgCache[key]) {{ const im = new Image(); im.src = IMAGES[camId][idx]; imgCache[key] = im; }}
  return imgCache[key];
}}
(function preload() {{
  for (let i = 0; i < Math.min(4, FRAMES.length); i++)
    CAM_IDS.forEach(c => getImg(c, i));
}})();

function drawCanvas(camId, frameIdx) {{
  const canvas = canvases[camId];
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const dw  = canvas.offsetWidth  || IMG_DIMS.disp_w;
  const dh  = canvas.offsetHeight || IMG_DIMS.disp_h;
  canvas.width = dw; canvas.height = dh;
  const img    = getImg(camId, frameIdx);
  const scaleX = dw / IMG_DIMS.native_w;
  const scaleY = dh / IMG_DIMS.native_h;
  function draw() {{
    ctx.clearRect(0, 0, dw, dh);
    ctx.drawImage(img, 0, 0, dw, dh);
    const frame = FRAMES[frameIdx];
    const bbox  = (DETECTIONS[frame] || {{}})[camId];
    if (bbox) {{
      const x1 = bbox[0]*scaleX, y1 = bbox[1]*scaleY;
      const x2 = bbox[2]*scaleX, y2 = bbox[3]*scaleY;
      ctx.strokeStyle = '#44aaff'; ctx.lineWidth = Math.max(2, dw*0.003);
      ctx.strokeRect(x1, y1, x2-x1, y2-y1);
    }}
  }}
  if (img.complete) draw(); else img.onload = draw;
}}

function update3D(frameIdx) {{
  const frame = FRAMES[frameIdx].toString();
  const pos   = POSITIONS[frame];

  // Hip sphere (trace 2)
  Plotly.restyle('plot3d', {{
    x: [pos ? [pos[0]] : [null]],
    y: [pos ? [pos[1]] : [null]],
    z: [pos ? [pos[2]] : [null]],
  }}, [2]);

  // Trajectory (trace 3) — last TRAJ_LEN positions
  if (pos) {{ posHistory.push(pos); if (posHistory.length > TRAJ_LEN) posHistory.shift(); }}
  const n  = posHistory.length;
  const tc = posHistory.map((_, i) => i / Math.max(n - 1, 1));
  Plotly.restyle('plot3d', {{
    x: [posHistory.map(p => p[0])],
    y: [posHistory.map(p => p[1])],
    z: [posHistory.map(p => p[2])],
    'marker.color': [tc],
  }}, [3]);

  // Overlay
  document.getElementById('ov-frame').textContent  = frame;
  const h = HEIGHTS[frame];
  document.getElementById('ov-height').textContent = h != null ? h.toFixed(0) : '—';
}}

function setFrame(idx) {{
  CAM_IDS.forEach(c => drawCanvas(c, idx));
  update3D(idx);
  document.getElementById('frame-label').textContent =
    'frame ' + FRAMES[idx] + ' (' + (idx+1) + '/' + FRAMES.length + ')';
  document.getElementById('slider').value = idx;
  for (let i = idx+1; i < Math.min(idx+4, FRAMES.length); i++)
    CAM_IDS.forEach(c => getImg(c, i));
}}

const slider = document.getElementById('slider');
slider.addEventListener('input', () => {{ posHistory = []; setFrame(+slider.value); }});

let playTimer = null, playIdx = 0;
document.getElementById('btnPlay').addEventListener('click', () => {{
  if (playTimer) return;
  if (playIdx >= FRAMES.length - 1) {{ playIdx = 0; posHistory = []; }}
  playTimer = setInterval(() => {{
    playIdx++;
    setFrame(playIdx);
    if (playIdx >= FRAMES.length - 1) {{ clearInterval(playTimer); playTimer = null; }}
  }}, 80);
}});
document.getElementById('btnPause').addEventListener('click', () => {{
  clearInterval(playTimer); playTimer = null; playIdx = +slider.value;
}});

setFrame(0);
window.addEventListener('resize', () => setFrame(+slider.value));
</script>
</body>
</html>"""

    VIEWER_HTML.write_text(html, encoding="utf-8")
    size_mb = VIEWER_HTML.stat().st_size / 1e6
    print(f"  Written → {VIEWER_HTML}  ({size_mb:.1f} MB)")
    return VIEWER_HTML


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 9 — Update config
# ═══════════════════════════════════════════════════════════════════════════════

def stage9_update_config(scale, n_samples, median_ha, shoulder_w, final_reproj):
    _header(9, "UPDATE CONFIG")
    if not CONFIG_JSON.exists():
        print(f"  {CONFIG_JSON} not found — skip")
        return
    cfg = json.loads(CONFIG_JSON.read_text())
    sw_mm = float(np.median(shoulder_w)) * scale if shoulder_w else 0.0
    cfg.setdefault("pricab_human_test", {})["scale_recovery"] = {
        "method":                                      "person height normalization",
        "person_height_mm":                            PERSON_HEIGHT_MM,
        "scale_factor":                                round(scale, 6),
        "samples_used":                                n_samples,
        "median_triangulated_height_mm_before_scaling": round(median_ha, 4),
        "median_triangulated_height_mm_after_scaling":  round(median_ha * scale, 1),
        "shoulder_width_validation_mm":                round(sw_mm, 1),
        "status":                                      "DONE",
    }
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"  Updated → {CONFIG_JSON}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       PriCaB — Scale Recovery Pipeline                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Person height: {PERSON_HEIGHT_MM:.0f} mm")

    cam_ids, cams, detections, ref_cam, reproj_orig = stage1_load()
    valid_frames, kp_thresh = stage2_find_valid_frames(cam_ids, detections)
    head_ankle, shoulder_w, hip_w, per_frame_heights = stage3_triangulate(
        cam_ids, cams, detections, valid_frames, kp_thresh)
    scale, median_ha, n_samples = stage4_compute_scale(head_ankle, shoulder_w)
    cams_s = stage5_apply_scale(cam_ids, cams, scale, median_ha, n_samples,
                                 head_ankle, shoulder_w, hip_w)
    final_reproj = stage6_validate(cam_ids, cams_s, detections, valid_frames)
    stage7_write_yamls(cam_ids, cams_s)
    viewer = stage8_build_viewer(cam_ids, cams_s, detections, scale, per_frame_heights)
    stage9_update_config(scale, n_samples, median_ha, shoulder_w, final_reproj)

    print(f"\n{'═'*60}")
    print(f"  Scale factor applied : {scale:.4f}  (1 unit = {1000/scale:.1f} mm)")
    print(f"  Final reproj error   : {final_reproj:.1f} px")
    print(f"  Viewer               : {viewer}")
    print('═'*60)


if __name__ == "__main__":
    main()
