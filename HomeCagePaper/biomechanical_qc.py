#!/usr/bin/env python3
"""
HomeCagePaper · biomechanical_qc.py

Biomechanical plausibility detector. Walks every session under data/sessions/,
triangulates per-keypoint 3D positions from the 4-camera detections, and flags
frames where the skeleton / motion is not physically plausible.

Checks performed
────────────────
 3D (on triangulated keypoints, in mm):
   · bone_length        per animal, per bone: |len - median| / MAD > BONE_MAD_K
   · bone_symmetry      left/right bone-length ratio outside [1/SYM_RATIO, SYM_RATIO]
   · keypoint_velocity  |kp(t) - kp(t-dt)| / dt > VEL_MAX_MM_PER_S   (dt from frame gap)
   · out_of_room        any kp outside ROOM_BOUNDS_MM (sanity — triangulation blew up)

 2D (per-camera bbox stream, before triangulation):
   · identity_jump      bbox centroid of same monkey_name jumps > ID_JUMP_PX
                        between consecutive frames (likely tracker swap)
   · identity_swap      bbox of monkey_name at frame t overlaps (IoU > IOU_SWAP)
                        the OTHER monkey's bbox at frame t-1 more than its own

Outputs (written to curated/plausibility/)
──────────────────────────────────────────
   violations_3d.parquet    one row per (session, frame, animal, check, detail)
   violations_2d.parquet    one row per (session, cam, frame, animal, check, detail)
   bone_stats.parquet       per (session, animal, bone) median / MAD / n
   summary.csv              violation counts per session / check / animal

Run:
    python3 HomeCagePaper/biomechanical_qc.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse the parsing + triangulation from curate_data.py so we stay consistent
from curate_data import (
    SESSIONS_ROOT, CALIBRATIONS, CURATED,
    ANIMALS, KP_NAMES, KP_CONF_PRED, MM_TO_CM,
    parse_full_session, projection_matrices, triangulate_session,
    discover_sessions,
)

# ─── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = CURATED / "plausibility"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_PATH = CALIBRATIONS / "human_pose.json"

# ─── Skeleton ───────────────────────────────────────────────────────────────
# Only body keypoints used by the 3D analyses (match curate_data.KEPT_KP_IDX):
#   5 L_shoulder  6 R_shoulder  11 L_hip  12 R_hip  13 L_knee  14 R_knee
KP = {n: i for i, n in enumerate(KP_NAMES)}

BONES = [
    # (name, kp_a, kp_b)
    ("shoulders",     KP["left_shoulder"], KP["right_shoulder"]),
    ("hips",          KP["left_hip"],      KP["right_hip"]),
    ("torso_left",    KP["left_shoulder"], KP["left_hip"]),
    ("torso_right",   KP["right_shoulder"],KP["right_hip"]),
    ("upper_leg_L",   KP["left_hip"],      KP["left_knee"]),
    ("upper_leg_R",   KP["right_hip"],     KP["right_knee"]),
]

SYMMETRY_PAIRS = [
    ("torso",     "torso_left",   "torso_right"),
    ("upper_leg", "upper_leg_L",  "upper_leg_R"),
]

# Keypoints we actually check for velocity / bounds (body only)
BODY_KPS = sorted({a for _, a, _ in BONES} | {b for _, _, b in BONES})

# ─── Thresholds ─────────────────────────────────────────────────────────────
# 3D
BONE_MAD_K       = 5.0          # flag when |len - median| > K * 1.4826 * MAD
SYM_RATIO        = 1.5          # max allowed L/R ratio (or R/L); flag if outside
VEL_MAX_MM_PER_S = 6000.0       # 6 m/s — sprint-level; anything above is triangulation noise
FPS_DEFAULT      = 30.0         # frame rate assumption for velocity check

# Room bounds in mm (curate_data uses cm; convert)
ROOM_BOUNDS_MM = {"x": (-500, 2400), "y": (-500, 3600), "z": (-300, 3500)}

# 2D
ID_JUMP_PX  = 800.0   # bbox-centroid jump between consecutive frames for same name
IOU_SWAP    = 0.30    # IoU threshold to call a swap "possible"

# ─── Helpers ────────────────────────────────────────────────────────────────
def mad(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def bbox_center(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def bbox_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    aB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


# ─── 3D violations ──────────────────────────────────────────────────────────
def compute_bone_series(kps_3d, animal):
    """Return dict bone_name -> DataFrame(frame, length_mm) for one animal."""
    per_bone = {name: [] for name, _, _ in BONES}
    for (frame, who), kp_dict in kps_3d.items():
        if who != animal:
            continue
        for name, a, b in BONES:
            pa = kp_dict.get(a); pb = kp_dict.get(b)
            if pa is None or pb is None:
                continue
            d = float(np.linalg.norm(np.asarray(pa) - np.asarray(pb)))
            per_bone[name].append((int(frame), d))
    return {name: pd.DataFrame(v, columns=["frame", "length_mm"]).sort_values("frame")
            for name, v in per_bone.items()}


def flag_bone_length(session, animal, bone_series):
    """Per-bone MAD-outlier flags + per-bone reference stats."""
    rows_viol, rows_stat = [], []
    for name, df in bone_series.items():
        if df.empty:
            rows_stat.append({
                "session": session, "animal": animal, "bone": name,
                "n": 0, "median_mm": np.nan, "mad_mm": np.nan,
            })
            continue
        med = float(df["length_mm"].median())
        m   = mad(df["length_mm"].values)
        rows_stat.append({
            "session": session, "animal": animal, "bone": name,
            "n": int(len(df)), "median_mm": med, "mad_mm": m,
        })
        if not np.isfinite(m) or m == 0:
            continue
        thr = BONE_MAD_K * 1.4826 * m
        bad = df[np.abs(df["length_mm"] - med) > thr]
        for _, r in bad.iterrows():
            rows_viol.append({
                "session": session, "frame": int(r["frame"]), "animal": animal,
                "check": "bone_length", "detail": name,
                "value": float(r["length_mm"]),
                "ref": med, "severity": float(abs(r["length_mm"] - med) / (1.4826 * m)),
            })
    return rows_viol, rows_stat


def flag_bone_symmetry(session, animal, bone_series):
    rows = []
    for pair_name, left, right in SYMMETRY_PAIRS:
        dfL, dfR = bone_series.get(left), bone_series.get(right)
        if dfL is None or dfR is None or dfL.empty or dfR.empty:
            continue
        m = dfL.merge(dfR, on="frame", suffixes=("_L", "_R"))
        if m.empty:
            continue
        ratio = m["length_mm_L"] / m["length_mm_R"].replace(0, np.nan)
        bad = m[(ratio > SYM_RATIO) | (ratio < 1 / SYM_RATIO)].copy()
        bad["ratio"] = ratio[bad.index]
        for _, r in bad.iterrows():
            rows.append({
                "session": session, "frame": int(r["frame"]), "animal": animal,
                "check": "bone_symmetry", "detail": pair_name,
                "value": float(r["ratio"]),
                "ref": 1.0,
                "severity": float(max(r["ratio"], 1 / r["ratio"])),
            })
    return rows


def flag_velocity_and_bounds(session, animal, kps_3d):
    """Frame-to-frame per-keypoint velocity + room-bounds sanity."""
    rows = []
    # Gather per-kp time series
    per_kp = {k: [] for k in BODY_KPS}
    for (frame, who), kp_dict in kps_3d.items():
        if who != animal:
            continue
        for k in BODY_KPS:
            p = kp_dict.get(k)
            if p is None:
                continue
            per_kp[k].append((int(frame), np.asarray(p, dtype=float)))
            # bounds check
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            if not (ROOM_BOUNDS_MM["x"][0] <= x <= ROOM_BOUNDS_MM["x"][1] and
                    ROOM_BOUNDS_MM["y"][0] <= y <= ROOM_BOUNDS_MM["y"][1] and
                    ROOM_BOUNDS_MM["z"][0] <= z <= ROOM_BOUNDS_MM["z"][1]):
                rows.append({
                    "session": session, "frame": int(frame), "animal": animal,
                    "check": "out_of_room", "detail": KP_NAMES[k],
                    "value": float(np.linalg.norm(p)),
                    "ref": np.nan, "severity": 1.0,
                })

    for k, series in per_kp.items():
        if len(series) < 2:
            continue
        series.sort(key=lambda t: t[0])
        for (f0, p0), (f1, p1) in zip(series[:-1], series[1:]):
            df = f1 - f0
            if df <= 0:
                continue
            dt = df / FPS_DEFAULT
            v = float(np.linalg.norm(p1 - p0) / dt)   # mm/s
            if v > VEL_MAX_MM_PER_S:
                rows.append({
                    "session": session, "frame": int(f1), "animal": animal,
                    "check": "keypoint_velocity", "detail": KP_NAMES[k],
                    "value": v, "ref": VEL_MAX_MM_PER_S,
                    "severity": float(v / VEL_MAX_MM_PER_S),
                })
    return rows


# ─── 2D identity checks ─────────────────────────────────────────────────────
def read_bbox_stream(session):
    """Return DataFrame(session, cam, frame, animal, bbox_x1..y2) for one session."""
    import csv as _csv
    rows = []
    sess_dir = SESSIONS_ROOT / session
    for cam in (102, 108, 113, 117):
        p = sess_dir / f"detections_cam{cam}.txt"
        if not p.exists():
            continue
        with open(p) as f:
            for r in _csv.DictReader(f):
                if r.get("monkey_name") not in ANIMALS:
                    continue
                try:
                    rows.append({
                        "cam":     int(cam),
                        "frame":   int(r["frame_number"]),
                        "animal":  r["monkey_name"],
                        "bbox_x1": float(r["bbox_x1"]),
                        "bbox_y1": float(r["bbox_y1"]),
                        "bbox_x2": float(r["bbox_x2"]),
                        "bbox_y2": float(r["bbox_y2"]),
                    })
                except (KeyError, ValueError):
                    continue
    return pd.DataFrame(rows)


def flag_identity_2d(session, bbox_df):
    """Identity jump + identity swap per camera."""
    rows = []
    for cam, cam_df in bbox_df.groupby("cam"):
        # Need per-frame dict of {animal: bbox} to check swaps
        frames = cam_df.groupby("frame")
        prev_frame, prev_boxes = None, {}
        for frame, g in frames:
            boxes = {r["animal"]: (r["bbox_x1"], r["bbox_y1"],
                                    r["bbox_x2"], r["bbox_y2"])
                     for _, r in g.iterrows()}
            if prev_frame is not None and (frame - prev_frame) <= 2:
                for name, box in boxes.items():
                    cx, cy   = bbox_center(box)
                    if name in prev_boxes:
                        pcx, pcy = bbox_center(prev_boxes[name])
                        jump = float(np.hypot(cx - pcx, cy - pcy))
                        if jump > ID_JUMP_PX:
                            rows.append({
                                "session": session, "cam": int(cam),
                                "frame": int(frame), "animal": name,
                                "check": "identity_jump", "detail": "bbox_center_px",
                                "value": jump, "ref": ID_JUMP_PX,
                                "severity": jump / ID_JUMP_PX,
                            })
                    # swap check: does this box match the OTHER animal's prev box better?
                    others = [o for o in prev_boxes if o != name]
                    if others and name in prev_boxes:
                        iou_self  = bbox_iou(box, prev_boxes[name])
                        iou_other = max(bbox_iou(box, prev_boxes[o]) for o in others)
                        if iou_other > IOU_SWAP and iou_other > iou_self:
                            rows.append({
                                "session": session, "cam": int(cam),
                                "frame": int(frame), "animal": name,
                                "check": "identity_swap", "detail": "iou_vs_other",
                                "value": float(iou_other),
                                "ref": float(iou_self),
                                "severity": float(iou_other - iou_self),
                            })
            prev_frame, prev_boxes = frame, boxes
    return rows


# ─── Pipeline ───────────────────────────────────────────────────────────────
def process_session(session, Ps):
    print(f"\n── {session} ──")
    sess_dir = SESSIONS_ROOT / session
    per_cam = parse_full_session(sess_dir, KP_CONF_PRED)
    if not per_cam:
        print("  no detection files, skipping")
        return [], [], [], []
    print(f"  cams: {sorted(per_cam)}  ·  dets: {sum(len(v) for v in per_cam.values())}")

    # 3D triangulation (reuse curate_data)
    kps_3d = triangulate_session(per_cam, Ps)

    viol3d, stats = [], []
    for animal in ANIMALS:
        bone_series = compute_bone_series(kps_3d, animal)
        v1, s1 = flag_bone_length(session, animal, bone_series)
        v2     = flag_bone_symmetry(session, animal, bone_series)
        v3     = flag_velocity_and_bounds(session, animal, kps_3d)
        viol3d += v1 + v2 + v3
        stats  += s1

    # 2D identity
    bbox_df = read_bbox_stream(session)
    viol2d = flag_identity_2d(session, bbox_df) if not bbox_df.empty else []

    n3 = len(viol3d); n2 = len(viol2d)
    print(f"  3D violations: {n3:>6}  ·  2D violations: {n2:>6}")
    return viol3d, viol2d, stats, [session]


def main():
    print(f"Loading calibration: {CALIB_PATH.name}")
    calib = json.loads(CALIB_PATH.read_text())
    Ps = projection_matrices(calib)
    print(f"  cameras placed: {sorted(Ps)}")

    sessions = discover_sessions()
    if not sessions:
        print("no sessions under data/sessions/")
        return

    all_v3, all_v2, all_stats = [], [], []
    for s in sessions:
        v3, v2, stats, _ = process_session(s, Ps)
        all_v3 += v3; all_v2 += v2; all_stats += stats

    df3 = pd.DataFrame(all_v3)
    df2 = pd.DataFrame(all_v2)
    dfs = pd.DataFrame(all_stats)

    (OUT_DIR / "violations_3d.parquet").unlink(missing_ok=True)
    (OUT_DIR / "violations_2d.parquet").unlink(missing_ok=True)
    (OUT_DIR / "bone_stats.parquet").unlink(missing_ok=True)

    if not df3.empty:
        df3.to_parquet(OUT_DIR / "violations_3d.parquet", index=False)
    if not df2.empty:
        df2.to_parquet(OUT_DIR / "violations_2d.parquet", index=False)
    if not dfs.empty:
        dfs.to_parquet(OUT_DIR / "bone_stats.parquet", index=False)

    # Summary: count violations per session / check / animal
    summary_rows = []
    if not df3.empty:
        g = df3.groupby(["session", "check", "animal"]).size().reset_index(name="n")
        g["layer"] = "3D"
        summary_rows.append(g)
    if not df2.empty:
        g = df2.groupby(["session", "check", "animal"]).size().reset_index(name="n")
        g["layer"] = "2D"
        summary_rows.append(g)
    if summary_rows:
        summary = pd.concat(summary_rows, ignore_index=True)
        summary = summary[["layer", "session", "check", "animal", "n"]]\
            .sort_values(["session", "layer", "check", "animal"])
        summary.to_csv(OUT_DIR / "summary.csv", index=False)
    else:
        summary = pd.DataFrame()

    print("\n══ Done ══")
    print(f"  3D violations: {len(df3)}")
    print(f"  2D violations: {len(df2)}")
    print(f"  bone stats rows: {len(dfs)}")
    print(f"  outputs → {OUT_DIR}")
    if not summary.empty:
        print("\nSummary (violations per session/check/animal):")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
