#!/usr/bin/env python3
"""
HomeCagePaper · curate_data.py

One-shot data curation. Reads the raw CSVs (GT and RT predictions) for every
session under data/ and the calibration JSON, then triangulates and writes a
single HDF5 store (curated/master.h5) containing two named tables:

    /keypoints   per (session, model, cam, frame, animal, kp_idx)
                 — validation set only (small, ~26 K rows)
                 — used by validation.py (GT vs RT comparison)

    /centroids   per (session, frame, animal)
                 — full RT inferences across every day under data/sessions/
                 — trunk centroid + size, no per-keypoint detail
                 — used by spatial_analysis.py and behavioral_tracking.py

This is the single source of truth for every figure in every section script.

Run:
    python3 HomeCagePaper/curate_data.py
"""
import sys
import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────────────────────
HERE     = Path(__file__).parent
DATA_DIR = HERE / "_data"
CURATED  = HERE / "curated"
CURATED.mkdir(parents=True, exist_ok=True)

VALIDATION_ROOT = DATA_DIR / "validation"
SESSIONS_ROOT   = DATA_DIR / "sessions"
CALIBRATIONS    = DATA_DIR / "calibrations"

# Room geometry — used to drop pathological triangulations
ROOM_BOUNDS_CM = {"x": (-50, 240), "y": (-50, 360), "z": (-30, 350)}

GT_DIR      = VALIDATION_ROOT / "ground_truth_eval"   # evaluation subset (RT vs human comparison)
GT_DENSE    = VALIDATION_ROOT / "ground_truth"        # denser GT for behaviour-on-GT analyses
PRED_DIR    = VALIDATION_ROOT / "RT_predictions"
CALIB_PATH  = CALIBRATIONS / "human_pose.json"

# Validation set: only the two sessions where we have human GT to score against
VALIDATION_SESSIONS = ["250708", "250715"]

# Full RT sessions: every day under data/sessions/ with detection_cam*.txt files
def discover_sessions():
    if not SESSIONS_ROOT.exists():
        return []
    return sorted(p.name for p in SESSIONS_ROOT.iterdir()
                  if p.is_dir() and any(p.glob("detections_cam*.txt")))
ANIMALS  = ["Elm", "Jok"]
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
# Body keypoints — face / wrists / ankles excluded
KEPT_KP_IDX = [5, 6, 11, 12, 13, 14]   # torso (shoulders + hips) + upper legs (knees)

KP_CONF_GT   = 0.0    # GT: presence == valid
KP_CONF_PRED = 0.30   # RT: confidence threshold

MM_TO_CM = 0.1


# ─── 1. CSV parsing ─────────────────────────────────────────────────────────
def parse_csv(path, conf_thresh):
    """Return {(frame, monkey): [17 × (x, y, conf)]}."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as f:
        for row in csv.DictReader(f):
            frame  = int(row["frame_number"])
            monkey = row["monkey_name"]
            kps = []
            ok = False
            for name in KP_NAMES:
                sx = row.get(f"{name}_x", "")
                sy = row.get(f"{name}_y", "")
                sc = row.get(f"{name}_confidence", "")
                if sx.strip() and sy.strip():
                    c = float(sc) if sc.strip() else 1.0
                    if c >= conf_thresh:
                        kps.append([float(sx), float(sy), c]); ok = True
                    else:
                        kps.append([0.0, 0.0, 0.0])
                else:
                    kps.append([0.0, 0.0, 0.0])
            if ok:
                out[(frame, monkey)] = kps
    return out


def parse_session(prefix, session, root, conf):
    """Return {cam: {(frame, monkey): kps}} for one validation session."""
    out = {}
    for cam_id in (102, 108, 113, 117):
        path = Path(root) / f"{prefix}_{session}_cam{cam_id}.csv"
        d = parse_csv(path, conf)
        if d:
            out[cam_id] = d
    return out


def parse_full_session(session_dir, conf):
    """Return {cam: {(frame, monkey): kps}} for a full RT session
    (data/sessions/<date>/detections_cam<id>.txt)."""
    out = {}
    for cam_id in (102, 108, 113, 117):
        path = Path(session_dir) / f"detections_cam{cam_id}.txt"
        d = parse_csv(path, conf)
        if d:
            out[cam_id] = d
    return out


# ─── 2. Calibration + triangulation ─────────────────────────────────────────
def load_calibration():
    return json.loads(CALIB_PATH.read_text())


def projection_matrices(calib):
    Ps = {}
    for cam_id, info in calib["cameras"].items():
        if not info.get("placed") or "R_world" not in info:
            continue
        K = np.array(info["K"])
        R = np.array(info["R_world"])
        T = np.array(info["T_world"]).reshape(3, 1)
        Ps[int(cam_id)] = K @ np.hstack([R, T])
    return Ps


def dlt_triangulate(Ps_list, pts_list):
    rows = []
    for P, p in zip(Ps_list, pts_list):
        rows += [p[0]*P[2] - P[0], p[1]*P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return (X[:3] / X[3]).tolist()


def triangulate_session(per_cam_dict, Ps):
    """Per-keypoint 3D triangulation across cameras.
    per_cam_dict: {cam: {(frame, monkey): kps}}.
    Returns {(frame, monkey): {kp_idx: [x, y, z] | None}}."""
    out = {}
    all_frames = set()
    for cam_data in per_cam_dict.values():
        all_frames.update(cam_data.keys())
    for key in all_frames:
        per_kp = {}
        for k in range(17):
            cam_Ps, cam_pts = [], []
            for cam, cam_data in per_cam_dict.items():
                if key not in cam_data:
                    continue
                kp = cam_data[key][k]
                if kp[2] > 0 and cam in Ps:
                    cam_Ps.append(Ps[cam])
                    cam_pts.append(kp[:2])
            if len(cam_Ps) >= 2:
                per_kp[k] = dlt_triangulate(cam_Ps, cam_pts)
            else:
                per_kp[k] = None
        out[key] = per_kp
    return out


def per_frame_animal_size_cm(kps_3d_dict):
    """3D bbox diagonal of the GT skeleton in cm."""
    pts = [kps_3d_dict[k] for k in KEPT_KP_IDX
           if k in kps_3d_dict and kps_3d_dict[k] is not None]
    if len(pts) < 2:
        return np.nan
    arr = np.asarray(pts, dtype=float)
    return float(np.linalg.norm(arr.max(0) - arr.min(0))) * MM_TO_CM


# ─── 3. Build the /keypoints table (validation only) ───────────────────────
def build_keypoints_table(Ps):
    rows = []
    for session in VALIDATION_SESSIONS:
        print(f"\n── Validation session {session} ──")
        gt_per_cam   = parse_session("gt",   session, GT_DIR,   KP_CONF_GT)
        pred_per_cam = parse_session("pred", session, PRED_DIR, KP_CONF_PRED)
        cams = sorted(set(gt_per_cam) | set(pred_per_cam))
        print(f"  cams: {cams}")

        # Eval universe = every (cam, frame) seen by any source
        cam_frames = {c: set() for c in cams}
        for source in (gt_per_cam, pred_per_cam):
            for c, cam_data in source.items():
                for (frame, _monkey) in cam_data:
                    cam_frames[c].add(int(frame))

        # 3D triangulation per source (one universe of (frame, monkey) per source)
        gt_3d   = triangulate_session(gt_per_cam,   Ps) if gt_per_cam   else {}
        pred_3d = triangulate_session(pred_per_cam, Ps) if pred_per_cam else {}

        # Per (frame, animal) GT animal size
        size_per = {}
        for (frame, monkey), kps3d in gt_3d.items():
            size_per[(int(frame), monkey)] = per_frame_animal_size_cm(kps3d)

        # Emit rows
        for cam in cams:
            for frame in sorted(cam_frames[cam]):
                for animal in ANIMALS:
                    # 2D detections in this cam, frame
                    gt_kps   = gt_per_cam.get(cam, {}).get((frame, animal))
                    pred_kps = pred_per_cam.get(cam, {}).get((frame, animal))
                    # 3D triangulations (denormalized — same value across cams)
                    gt_3d_kps   = gt_3d.get((frame, animal),   {})
                    pred_3d_kps = pred_3d.get((frame, animal), {})
                    asize       = size_per.get((frame, animal), np.nan)

                    for k in KEPT_KP_IDX:
                        for model_name, kps2d, kps3d in (
                            ("GT", gt_kps,   gt_3d_kps),
                            ("RT", pred_kps, pred_3d_kps),
                        ):
                            x2d = y2d = c2d = np.nan
                            if kps2d is not None and k < len(kps2d):
                                _x, _y, _c = kps2d[k]
                                if _c > 0:
                                    x2d, y2d, c2d = _x, _y, _c
                            x3d = y3d = z3d = np.nan
                            pt = kps3d.get(k) if isinstance(kps3d, dict) else None
                            if pt is not None:
                                x3d, y3d, z3d = pt
                            rows.append({
                                "session":           session,
                                "model":             model_name,
                                "cam":               cam,
                                "frame":             int(frame),
                                "animal":            animal,
                                "kp_idx":            int(k),
                                "kp_name":           KP_NAMES[k],
                                "x_2d":              x2d,
                                "y_2d":              y2d,
                                "conf_2d":           c2d,
                                "x_3d":              x3d,
                                "y_3d":              y3d,
                                "z_3d":              z3d,
                                "animal_size_cm":    asize,
                                "in_frame_universe": True,
                            })

    df = pd.DataFrame(rows)
    return df


# ─── 4. Build the /centroids table (full RT sessions) ──────────────────────
def build_centroids_table(Ps):
    """Walk every full session under data/sessions/, triangulate, emit one
    row per (session, frame, animal) with the RT trunk centroid in cm."""
    rows = []
    sessions = discover_sessions()
    if not sessions:
        print("  no full sessions under data/sessions/ — /centroids will be empty")
        return pd.DataFrame(columns=[
            "session", "frame", "animal",
            "x_cm", "y_cm", "z_cm",
            "n_kps", "size_cm",
        ])

    for session in sessions:
        sess_dir = SESSIONS_ROOT / session
        print(f"\n── Full session {session} ──")
        per_cam = parse_full_session(sess_dir, KP_CONF_PRED)
        if not per_cam:
            print("  no detection files, skipping")
            continue
        cams = sorted(per_cam)
        n_dets = sum(len(v) for v in per_cam.values())
        print(f"  cams: {cams}  ·  detections: {n_dets}")

        kps_3d = triangulate_session(per_cam, Ps)

        n_emit = 0
        for (frame, monkey), kps_dict in kps_3d.items():
            if monkey not in ANIMALS:
                continue
            valid_pts = [kps_dict[k] for k in KEPT_KP_IDX
                         if k in kps_dict and kps_dict[k] is not None]
            if len(valid_pts) < 2:
                continue
            arr = np.asarray(valid_pts, dtype=float)
            centroid = arr.mean(axis=0) * MM_TO_CM
            size_cm  = float(np.linalg.norm(arr.max(0) - arr.min(0))) * MM_TO_CM
            cx, cy, cz = float(centroid[0]), float(centroid[1]), float(centroid[2])
            # Drop centroids whose triangulation went outside the room bounds
            if not (ROOM_BOUNDS_CM["x"][0] <= cx <= ROOM_BOUNDS_CM["x"][1] and
                    ROOM_BOUNDS_CM["y"][0] <= cy <= ROOM_BOUNDS_CM["y"][1] and
                    ROOM_BOUNDS_CM["z"][0] <= cz <= ROOM_BOUNDS_CM["z"][1]):
                continue
            rows.append({
                "session": session,
                "frame":   int(frame),
                "animal":  monkey,
                "x_cm":    cx,
                "y_cm":    cy,
                "z_cm":    cz,
                "n_kps":   int(len(valid_pts)),
                "size_cm": size_cm,
            })
            n_emit += 1
        print(f"  → {n_emit} centroid rows")

    return pd.DataFrame(rows)


# ─── 5. Build the /detections table (per-camera presence in full sessions) ─
def build_detections_table():
    """One row per (session, cam, frame, animal) RT detection in the full
    sessions, including the 2D bounding box (x1, y1, x2, y2) so downstream
    analyses can do IoU-based track checks."""
    rows = []
    sessions = discover_sessions()
    for session in sessions:
        sess_dir = SESSIONS_ROOT / session
        per_cam = parse_full_session(sess_dir, KP_CONF_PRED)
        for cam, cam_data in per_cam.items():
            for (frame, monkey), kps in cam_data.items():
                if monkey not in ANIMALS:
                    continue
                # The kps list does not include the bbox; bbox lives in the
                # raw row. We re-read it from the source CSV here once per
                # session/cam to keep the parser simple.
                rows.append({
                    "session": session,
                    "cam":     int(cam),
                    "frame":   int(frame),
                    "animal":  monkey,
                })
    # Walk the source files once more to attach the bbox for every row
    bbox_index = {}
    for session in sessions:
        sess_dir = SESSIONS_ROOT / session
        for cam_id in (102, 108, 113, 117):
            path = sess_dir / f"detections_cam{cam_id}.txt"
            if not path.exists():
                continue
            import csv as _csv
            with open(path) as f:
                for row in _csv.DictReader(f):
                    monkey = row.get("monkey_name", "")
                    if monkey not in ANIMALS:
                        continue
                    try:
                        frame = int(row["frame_number"])
                        bbox  = (float(row["bbox_x1"]), float(row["bbox_y1"]),
                                 float(row["bbox_x2"]), float(row["bbox_y2"]))
                    except (KeyError, ValueError):
                        continue
                    bbox_index[(session, int(cam_id), frame, monkey)] = bbox
    df = pd.DataFrame(rows)
    if df.empty:
        for c in ("bbox_x1","bbox_y1","bbox_x2","bbox_y2"):
            df[c] = []
        return df
    bx = df.apply(lambda r: bbox_index.get(
        (r["session"], int(r["cam"]), int(r["frame"]), r["animal"]),
        (np.nan, np.nan, np.nan, np.nan)), axis=1)
    df["bbox_x1"] = [b[0] for b in bx]
    df["bbox_y1"] = [b[1] for b in bx]
    df["bbox_x2"] = [b[2] for b in bx]
    df["bbox_y2"] = [b[3] for b in bx]
    return df


# ─── 6. Save ────────────────────────────────────────────────────────────────
def main():
    print(f"Loading calibration: {CALIB_PATH.name}")
    calib = load_calibration()
    Ps = projection_matrices(calib)
    print(f"  cameras placed: {sorted(Ps)}")

    print("\n══ Building /keypoints table (validation) ══")
    kp_df = build_keypoints_table(Ps)

    print("\n══ Building /centroids table (full RT sessions) ══")
    ctr_df = build_centroids_table(Ps)

    print("\n══ Building /detections table (per-cam presence) ══")
    det_df = build_detections_table()
    print(f"  rows: {len(det_df)}")

    out = CURATED / "master.h5"
    if out.exists():
        out.unlink()

    # HDF5 with two named tables, queryable on the listed data_columns
    with pd.HDFStore(str(out), mode="w", complevel=9, complib="blosc") as store:
        store.put("keypoints", kp_df,
                  format="table",
                  data_columns=["session", "model", "cam", "animal", "kp_idx"],
                  min_itemsize={"session": 16, "model": 4, "animal": 8,
                                "kp_name": 32})
        store.put("centroids", ctr_df,
                  format="table",
                  data_columns=["session", "animal"],
                  min_itemsize={"session": 16, "animal": 8})
        store.put("detections", det_df,
                  format="table",
                  data_columns=["session", "cam", "animal", "frame"],
                  min_itemsize={"session": 16, "animal": 8})

    print(f"\n── master.h5 ──")
    print(f"  /keypoints  : {len(kp_df):>7} rows  ·  cols: {list(kp_df.columns)}")
    print(f"  /centroids  : {len(ctr_df):>7} rows  ·  cols: {list(ctr_df.columns)}")
    print(f"  /detections : {len(det_df):>7} rows  ·  cols: {list(det_df.columns)}")
    if not ctr_df.empty:
        print(f"  centroid sessions: {sorted(ctr_df['session'].unique())}")
    print(f"\nSaved → {out}  ({out.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
