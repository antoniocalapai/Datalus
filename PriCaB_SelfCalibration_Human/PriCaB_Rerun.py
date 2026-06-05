#!/usr/bin/env python3
"""
PriCaB_Rerun.py  —  PriCaB extrinsics via maximum-spanning-tree + wall alignment
==================================================================================

Approach (no hub camera):
  Step 1  Co-detection matrix  (bbox_conf > 0.5 both cameras)
  Step 2  Maximum spanning tree (Kruskal's)
  Step 3  Reference = highest total co-detections; BFS traversal with
          essmat+recoverPose against best already-placed neighbour
  Step 4  Any camera with <30 co-det with all placed cameras → UNSOLVABLE
  Step 5  Scale from person height (1700 mm) + wall constraint alignment
  Step 6  Validation table; STOP

Usage:
    python3 PriCaB_Rerun.py
"""

import json
import re
from collections import deque
from datetime import datetime
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize as sp_minimize

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
_REPO           = BASE_DIR.parent
OUT_DIR         = BASE_DIR / "PriCaB_output"
INTRINSICS_NPZ  = _REPO / "_data" / "DatalusCalibration" / "intrinsics.npz"
STABLE_YAML_DIR = _REPO / "_archive" / "versions" / "v2_pricab_stable" / "yamls"
CONFIG_JSON     = _REPO / "_archive" / "datalus_config.json"
VIDEO_FOLDER    = _REPO / "_data" / "Measurements" / "250404_HumanTest_2"

POSES_NPZ  = OUT_DIR / "pricab_poses.npz"
SCALED_NPZ = OUT_DIR / "pricab_poses_scaled.npz"

# ── Constants ────────────────────────────────────────────────────────────────
PERSON_HEIGHT_MM  = 1700.0
SCALE_KPS         = [0, 5, 6, 11, 12, 15, 16]   # nose, l/r-sh, l/r-hip, l/r-ankle
KP_CONF_THRESH    = 0.5
CODET_BBOX_CONF   = 0.5    # for co-detection matrix
RANSAC_THRESH     = 0.001  # essmat normalised-coords threshold
MIN_CODET_PLACE   = 30     # min co-detections to attempt essmat placement
LENS_FOCAL_MM     = 8.0
SENSOR_WIDTH_MM   = 11.2
VIDEO_EXTS        = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}

# ── Wall groups ──────────────────────────────────────────────────────────────
WALL_A   = ["113", "106", "107", "105", "110"]
WALL_B   = ["118", "112", "109", "114", "102"]
WALL_C   = ["106", "110", "112", "118", "119"]
WALL_D   = ["102", "109", "105", "107", "111"]
GROUND   = ["106", "107", "112", "109"]
CEILING  = ["110", "105", "102", "118"]
MID_Z    = ["113", "119", "114", "111"]
TOP      = ["103"]


def _hdr(n, title):
    print(f"\n{'═'*62}")
    print(f"STEP {n} — {title}")
    print('═'*62)


# ════════════════════════════════════════════════════════════════════════════
# Part 3 — FPS
# ════════════════════════════════════════════════════════════════════════════

def read_fps_and_cam_info():
    _hdr(3, "READ VIDEO FPS FROM METADATA")
    videos  = sorted(p for p in VIDEO_FOLDER.iterdir() if p.suffix in VIDEO_EXTS)
    cam_info = {}
    fps_map  = {}
    for vpath in videos:
        nums   = re.findall(r'\d+', vpath.stem)
        cam_id = next((n for n in reversed(nums) if len(n) == 3), None)
        if cam_id is None:
            continue
        cap  = cv2.VideoCapture(str(vpath))
        fps  = cap.get(cv2.CAP_PROP_FPS)
        w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nf   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps <= 0:
            fps = 25.0
        dur = nf / fps
        cam_info[cam_id] = {"path": vpath, "w": w, "h": h,
                             "fps": fps, "n_frames": nf, "duration_s": dur}
        fps_map[cam_id] = fps
        print(f"  cam {cam_id}: {w}×{h}  {fps:.3f} fps  {nf} fr  {dur:.1f}s")

    cfg = {}
    if CONFIG_JSON.exists():
        cfg = json.loads(CONFIG_JSON.read_text())
    cfg.setdefault("pricab_human_test", {})["camera_fps"] = {
        k: round(float(v), 6) for k, v in fps_map.items()}
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"  FPS → {CONFIG_JSON}")
    return cam_info


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _load_pose_files(cam_ids):
    detections = {}
    for cam_id in sorted(cam_ids):
        cam_dets = {}
        txt = OUT_DIR / f"pose_{cam_id}.txt"
        if txt.exists():
            for line in txt.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 57:
                    continue
                fidx      = int(cols[0])
                bbox      = [float(c) for c in cols[1:5]]
                bbox_conf = float(cols[5])
                kps       = np.array(cols[6:57], dtype=np.float32).reshape(17, 3)
                cam_dets[fidx] = {"bbox": bbox, "bbox_conf": bbox_conf, "kps": kps}
        detections[cam_id] = cam_dets
        print(f"  cam {cam_id}: {len(cam_dets)} detections")
    return detections


def _load_intrinsics(cam_ids, cam_info):
    """Priority: intrinsics.npz > stable YAMLs > thin-lens."""
    loaded_K, loaded_dist = {}, {}
    if INTRINSICS_NPZ.exists():
        npz         = np.load(str(INTRINSICS_NPZ))
        loaded_K    = {k[:-2]: npz[k] for k in npz.files if k.endswith("_K")}
        loaded_dist = {k[:-5]: npz[k] for k in npz.files if k.endswith("_dist")}

    yaml_K, yaml_dist = {}, {}
    if STABLE_YAML_DIR.exists():
        for yp in sorted(STABLE_YAML_DIR.glob("*.yaml")):
            cid = yp.stem
            try:
                fs  = cv2.FileStorage(str(yp), cv2.FILE_STORAGE_READ)
                KT  = fs.getNode("intrinsicMatrix").mat()
                dn  = fs.getNode("distortionCoefficients").mat()
                fs.release()
                if KT is not None and KT.shape == (3, 3):
                    yaml_K[cid]    = KT.T.astype(np.float64)
                    yaml_dist[cid] = (dn.ravel() if dn is not None
                                      else np.zeros(5)).astype(np.float64)
            except Exception:
                pass

    cams = {}
    for cam_id in sorted(cam_ids):
        w = cam_info[cam_id]["w"]; h = cam_info[cam_id]["h"]
        if cam_id in loaded_K:
            K = loaded_K[cam_id].astype(np.float64)
            d = loaded_dist.get(cam_id, np.zeros(5)).ravel().astype(np.float64)
            src = "intrinsics.npz"
        elif cam_id in yaml_K:
            K = yaml_K[cam_id]; d = yaml_dist[cam_id]
            src = "stable YAMLs"
        else:
            fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) * w
            K  = np.array([[fx,0,w/2.],[0,fx,h/2.],[0,0,1.]], dtype=np.float64)
            d  = np.zeros(5, dtype=np.float64)
            src = "thin-lens"
        cams[cam_id] = {"K": K, "dist": d, "w": w, "h": h}
        print(f"  cam {cam_id}: fx={K[0,0]:.1f}  cx={K[0,2]:.1f}  cy={K[1,2]:.1f}  [{src}]")
    return cams


# ════════════════════════════════════════════════════════════════════════════
# Steps 1–4: Spanning-tree extrinsics
# ════════════════════════════════════════════════════════════════════════════

def run_extrinsics_spanning_tree(cam_ids, detections, cams):
    cam_ids_s = sorted(cam_ids)

    # ── Step 1: Co-detection matrix ──────────────────────────────────────────
    _hdr(1, "CO-DETECTION MATRIX  (bbox_conf > 0.5 both cameras)")
    co = {ca: {cb: 0 for cb in cam_ids_s} for ca in cam_ids_s}
    for ca, cb in combinations(cam_ids_s, 2):
        n = sum(1 for f in sorted(set(detections[ca]) & set(detections[cb]))
                if detections[ca][f]["bbox_conf"] > CODET_BBOX_CONF
                and detections[cb][f]["bbox_conf"] > CODET_BBOX_CONF)
        co[ca][cb] = co[cb][ca] = n

    # Print matrix
    print(f"\n  {'':>4}", end="")
    for c in cam_ids_s:
        print(f" {c:>4}", end="")
    print()
    for ca in cam_ids_s:
        print(f"  {ca:>4}", end="")
        for cb in cam_ids_s:
            if ca == cb:
                print(f" {'—':>4}", end="")
            else:
                print(f" {co[ca][cb]:>4}", end="")
        print()

    totals = {c: sum(co[c][o] for o in cam_ids_s if o != c) for c in cam_ids_s}
    print(f"\n  Row totals:")
    for c in sorted(cam_ids_s, key=lambda x: -totals[x]):
        print(f"    cam {c}: {totals[c]}")

    # ── Step 2: Maximum spanning tree (Kruskal's) ────────────────────────────
    _hdr(2, "MAXIMUM SPANNING TREE  (Kruskal's)")
    edges = sorted([(co[ca][cb], ca, cb)
                    for ca, cb in combinations(cam_ids_s, 2)],
                   reverse=True)

    par = {c: c for c in cam_ids_s}
    rnk = {c: 0 for c in cam_ids_s}

    def _find(x):
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x

    def _union(x, y):
        px, py = _find(x), _find(y)
        if px == py:
            return False
        if rnk[px] < rnk[py]:
            px, py = py, px
        par[py] = px
        if rnk[px] == rnk[py]:
            rnk[px] += 1
        return True

    mst = []
    for w, ca, cb in edges:
        if _union(ca, cb):
            mst.append((w, ca, cb))
        if len(mst) == len(cam_ids_s) - 1:
            break

    print(f"\n  MST edges ({len(mst)}):")
    for w, ca, cb in sorted(mst, reverse=True):
        print(f"    {ca}↔{cb}: {w} co-detections")

    adj = {c: [] for c in cam_ids_s}
    for w, ca, cb in mst:
        adj[ca].append((cb, w))
        adj[cb].append((ca, w))

    # ── Step 3: BFS from reference, essmat+recoverPose per edge ─────────────
    _hdr(3, "BFS TRAVERSAL — essmat+recoverPose")
    ref_cam = max(cam_ids_s, key=lambda c: totals[c])
    print(f"\n  Reference: cam {ref_cam}  (total co-det {totals[ref_cam]})")

    poses   = {ref_cam: {"R": np.eye(3, dtype=np.float64),
                          "T": np.zeros((3, 1), dtype=np.float64)}}
    queue   = deque([ref_cam])
    visited = {ref_cam}
    failed  = []
    placement_log = []   # (child, parent, co_det, kp_pairs, inliers, inl_pct)

    while queue:
        par_id = queue.popleft()
        for child_id, _ in adj[par_id]:
            if child_id in visited:
                continue
            visited.add(child_id)

            # Best already-placed neighbour for this child
            best_par  = max(poses.keys(), key=lambda p: co[child_id][p])
            best_codet = co[child_id][best_par]

            # Step 4: threshold
            if best_codet < MIN_CODET_PLACE:
                print(f"  cam {child_id}: UNSOLVABLE  "
                      f"(max co-det={best_codet} < {MIN_CODET_PLACE})")
                failed.append(child_id)
                continue

            # Collect KP correspondences (high-conf frames only)
            shared_frames = [
                f for f in sorted(set(detections[child_id]) & set(detections[best_par]))
                if detections[child_id][f]["bbox_conf"] > CODET_BBOX_CONF
                and detections[best_par][f]["bbox_conf"] > CODET_BBOX_CONF
            ]
            pts_a, pts_b = [], []
            for f in shared_frames:
                ka = detections[best_par][f]["kps"]
                kb = detections[child_id][f]["kps"]
                for ki in range(17):
                    if ka[ki, 2] >= KP_CONF_THRESH and kb[ki, 2] >= KP_CONF_THRESH:
                        pts_a.append([ka[ki, 0], ka[ki, 1]])
                        pts_b.append([kb[ki, 0], kb[ki, 1]])

            if len(pts_a) < 8:
                print(f"  cam {child_id}: < 8 kp pairs with cam {best_par} — FAILED")
                failed.append(child_id)
                continue

            pts_a = np.array(pts_a, dtype=np.float64)
            pts_b = np.array(pts_b, dtype=np.float64)
            Ka, da = cams[best_par]["K"],   cams[best_par]["dist"]
            Kb, db = cams[child_id]["K"],   cams[child_id]["dist"]

            n_a = cv2.undistortPoints(pts_a.reshape(-1,1,2), Ka, da).reshape(-1,2)
            n_b = cv2.undistortPoints(pts_b.reshape(-1,1,2), Kb, db).reshape(-1,2)

            try:
                E, mask_e = cv2.findEssentialMat(
                    n_a, n_b, np.eye(3),
                    method=cv2.RANSAC, threshold=RANSAC_THRESH, prob=0.999)
            except cv2.error as exc:
                print(f"  cam {child_id}: essmat cv2.error ({exc}) — FAILED")
                failed.append(child_id)
                continue

            if E is None or mask_e is None:
                print(f"  cam {child_id}: essmat returned None — FAILED")
                failed.append(child_id)
                continue

            inliers   = int(mask_e.ravel().sum())
            inl_pct   = 100.0 * inliers / len(pts_a)
            _, R_ba, T_ba, _ = cv2.recoverPose(E, n_a, n_b, np.eye(3), mask=mask_e)
            T_ba = T_ba.ravel()

            # Compose world-frame pose:
            # p_child = R_ba @ p_par + T_ba
            # p_par   = R_p  @ p_world + T_p
            # => R_child = R_ba @ R_p,  T_child = R_ba @ T_p + T_ba
            R_p = poses[best_par]["R"]
            T_p = poses[best_par]["T"].ravel()
            R_c = R_ba @ R_p
            T_c = (R_ba @ T_p + T_ba).reshape(3, 1)
            C_c = (-R_c.T @ T_c).ravel()

            poses[child_id] = {"R": R_c, "T": T_c}
            placement_log.append((child_id, best_par, best_codet,
                                   len(pts_a), inliers, inl_pct))
            print(f"  cam {child_id}: via cam {best_par}  "
                  f"({best_codet} co-det, {len(pts_a)} kp, "
                  f"{inliers} inl={inl_pct:.0f}%)  "
                  f"C=[{C_c[0]:.0f},{C_c[1]:.0f},{C_c[2]:.0f}]")
            queue.append(child_id)

    # Any still unvisited?
    for cid in cam_ids_s:
        if cid not in visited and cid not in failed:
            print(f"  cam {cid}: unreachable — FAILED")
            failed.append(cid)

    # Finalise C, P
    for cid in poses:
        R = poses[cid]["R"]; T = poses[cid]["T"]
        poses[cid]["C"] = (-R.T @ T).ravel()
        poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R, T])

    placed_ids = [c for c in cam_ids_s if c not in failed]
    print(f"\n  Placed: {len(placed_ids)}/{len(cam_ids_s)}  "
          f"failed: {failed if failed else 'none'}")

    # ── Step 5a: Scale from person height ────────────────────────────────────
    _hdr(5, "SCALE RECOVERY  (person height = 1700 mm)")

    def _tri(Pa, Pb, xa, ya, xb, yb):
        X4 = cv2.triangulatePoints(
            Pa, Pb,
            np.array([[xa],[ya]], dtype=np.float64),
            np.array([[xb],[yb]], dtype=np.float64))
        w = float(X4[3])
        if abs(w) < 1e-10:
            return None
        X3 = (X4[:3] / w).ravel()
        return X3 if np.all(np.isfinite(X3)) else None

    # Collect head-to-ankle distances from ALL pairs of placed cameras
    ha_all = []
    for ca2, cb2 in combinations(placed_ids, 2):
        Pa2 = poses[ca2]["P"]; Pb2 = poses[cb2]["P"]
        Ra2 = poses[ca2]["R"]; Ta2 = poses[ca2]["T"].ravel()
        Rb2 = poses[cb2]["R"]; Tb2 = poses[cb2]["T"].ravel()
        shared = [
            f for f in sorted(set(detections[ca2]) & set(detections[cb2]))
            if detections[ca2][f]["bbox_conf"] > CODET_BBOX_CONF
            and detections[cb2][f]["bbox_conf"] > CODET_BBOX_CONF
        ]
        for f in shared:
            kps_a = detections[ca2][f]["kps"]
            kps_b = detections[cb2][f]["kps"]
            ok = True
            pts3d = {}
            for ki in SCALE_KPS:
                if kps_a[ki,2] < KP_CONF_THRESH or kps_b[ki,2] < KP_CONF_THRESH:
                    ok = False; break
                X3 = _tri(Pa2, Pb2,
                           float(kps_a[ki,0]), float(kps_a[ki,1]),
                           float(kps_b[ki,0]), float(kps_b[ki,1]))
                if X3 is None:
                    ok = False; break
                if (Ra2 @ X3 + Ta2)[2] <= 0 or (Rb2 @ X3 + Tb2)[2] <= 0:
                    ok = False; break
                pts3d[ki] = X3
            if not ok:
                continue
            nose   = pts3d[0]
            ankl_m = (pts3d[15] + pts3d[16]) / 2
            ha     = float(np.linalg.norm(nose - ankl_m))
            if ha > 0:
                ha_all.append((ha, ca2, cb2, f))

    if not ha_all:
        print("  [WARN] No valid height measurements — scale=1.0")
        scale = 1.0
        med_raw = 0.0
    else:
        med_raw = float(np.median([h for h, *_ in ha_all]))
        scale   = PERSON_HEIGHT_MM / med_raw
        print(f"  Valid height measurements: {len(ha_all)}")
        print(f"  Median height (raw units): {med_raw:.6f}")
        print(f"  Scale factor:              {scale:.4f}")
        print(f"  Median height (after scale): {med_raw * scale:.1f} mm  "
              f"({'✓' if abs(med_raw*scale - PERSON_HEIGHT_MM) < 50 else '⚠'})")

    # Apply scale to all T vectors (unit-norm → mm)
    for cid in placed_ids:
        if cid == ref_cam:
            continue
        poses[cid]["T"] = poses[cid]["T"] * scale
        # Recompute C and P with scaled T
        R = poses[cid]["R"]; T = poses[cid]["T"]
        poses[cid]["C"] = (-R.T @ T).ravel()
        poses[cid]["P"] = cams[cid]["K"] @ np.hstack([R, T])

    # Recompute ref C (stays at 0)
    poses[ref_cam]["C"] = np.zeros(3)

    # Save pricab_poses.npz (pre-alignment, metric)
    save = {"reference_camera":      np.array([ref_cam]),
            "reprojection_error_px": np.array([0.0]),   # no BA in this approach
            "scale_factor":          np.array([scale])}
    for cid in cam_ids_s:
        if cid in poses:
            save[f"{cid}_R"] = poses[cid]["R"]
            save[f"{cid}_T"] = poses[cid]["T"]
        else:
            save[f"{cid}_R"] = np.eye(3, dtype=np.float64)
            save[f"{cid}_T"] = np.zeros((3,1), dtype=np.float64)
    np.savez(str(POSES_NPZ), **save)
    print(f"  Saved → {POSES_NPZ}")

    return poses, ref_cam, scale, med_raw, failed, co


# ════════════════════════════════════════════════════════════════════════════
# Step 5b: Wall constraint alignment
# ════════════════════════════════════════════════════════════════════════════

def apply_wall_alignment(cam_ids_s, poses, failed, cams):
    _hdr("7", "WALL CONSTRAINT ALIGNMENT")

    # Axis permutation to Z-up frame:
    # new_X = raw_X,  new_Y = raw_Z,  new_Z = -raw_Y
    A = np.array([[1., 0., 0.],
                  [0., 0., 1.],
                  [0.,-1., 0.]])

    placed_ids = [c for c in cam_ids_s if c not in failed]

    def _grp(lst):
        return [c for c in lst if c in placed_ids]

    G_A    = _grp(WALL_A)
    G_B    = _grp(WALL_B)
    G_C    = _grp(WALL_C)
    G_D    = _grp(WALL_D)
    G_gnd  = _grp(GROUND)
    G_ceil = _grp(CEILING)
    G_mid  = _grp(MID_Z)

    C_zu = {c: A @ poses[c]["C"] for c in placed_ids}

    def _Rz(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

    def _Rx(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])

    def _Ry(t):
        c, s = np.cos(t), np.sin(t)
        return np.array([[c,0.,s],[0.,1.,0.],[-s,0.,c]])

    # Step 1: Rz yaw — 72 multi-start
    def _yaw_loss(p):
        Rz = _Rz(float(p[0]) if hasattr(p, '__len__') else float(p))
        Cr = {c: Rz @ C_zu[c] for c in C_zu}
        loss = 0.0
        for grp, ax in [(G_A,0),(G_B,0),(G_C,1),(G_D,1)]:
            xs = [Cr[c][ax] for c in grp if c in Cr]
            if len(xs) >= 2:
                loss += float(np.var(xs))
        return loss

    best = None
    for t0 in np.linspace(-np.pi, np.pi, 72, endpoint=False):
        r = sp_minimize(_yaw_loss, [t0], method='Nelder-Mead',
                        options={'xatol':1e-9,'fatol':1e-9,'maxiter':3000})
        if best is None or r.fun < best.fun:
            best = r

    R_yaw   = _Rz(float(best.x[0]))
    C_yawed = {c: R_yaw @ C_zu[c] for c in C_zu}
    print(f"\n  Yaw (Rz): θ={np.degrees(best.x[0]):.2f}°  "
          f"loss={best.fun:.1f}  (72 starts)")

    # Step 2: Tilt (Rx, Ry) — minimise Z variance of Ground cameras
    def _tilt_loss(p):
        Rt = _Rx(p[0]) @ _Ry(p[1])
        Cr = {c: Rt @ C_yawed[c] for c in C_yawed}
        zs = [Cr[c][2] for c in G_gnd if c in Cr]
        return float(np.var(zs)) if len(zs) >= 2 else 0.0

    res_t  = sp_minimize(_tilt_loss,[0.,0.],method='Nelder-Mead',
                         options={'xatol':1e-7,'fatol':1e-7,'maxiter':4000})
    R_tilt = _Rx(float(res_t.x[0])) @ _Ry(float(res_t.x[1]))
    C_tild = {c: R_tilt @ C_yawed[c] for c in C_yawed}
    print(f"  Tilt (Rx,Ry): rx={np.degrees(res_t.x[0]):.2f}°  "
          f"ry={np.degrees(res_t.x[1]):.2f}°  ok={res_t.success}")

    # Step 3: Translate — Wall A ∩ Wall C corner (cam106/cam110 mean) → origin XY
    origin = [c for c in ["106","110"] if c in C_tild]
    if origin:
        cxy     = np.mean([C_tild[c][:2] for c in origin], axis=0)
        t_align = np.array([-cxy[0], -cxy[1], 0.])
    else:
        t_align = np.zeros(3)
        print("  [WARN] cam106/cam110 not placed — skipping XY translation")
    print(f"  Translate XY: ({t_align[0]:.1f}, {t_align[1]:.1f}) mm")

    R_part7 = R_tilt @ R_yaw
    R_total  = R_part7 @ A

    for cid in cam_ids_s:
        if cid in failed:
            continue
        C_raw   = poses[cid]["C"]
        C_zu_c  = A @ C_raw
        C_final = R_part7 @ C_zu_c + t_align
        R_new   = poses[cid]["R"] @ R_total.T
        T_new   = (-R_new @ C_final).reshape(3,1)
        poses[cid]["C_zu_raw"] = C_zu_c
        poses[cid]["C_aligned"] = C_final
        poses[cid]["R"]         = R_new
        poses[cid]["T"]         = T_new
        poses[cid]["P"]         = (cams[cid]["K"] @ np.hstack([R_new, T_new]))

    return poses, R_total, t_align


# ════════════════════════════════════════════════════════════════════════════
# Step 6: Validation table
# ════════════════════════════════════════════════════════════════════════════

def print_validation_table(cam_ids_s, poses, failed):
    _hdr(6, "VALIDATION TABLE")

    placed_ids = [c for c in cam_ids_s if c not in failed]
    wall_mem   = {}
    for gname, glist in [("A",WALL_A),("B",WALL_B),("C",WALL_C),("D",WALL_D),
                          ("Gnd",GROUND),("Ceil",CEILING),("Mid",MID_Z),("Top",TOP)]:
        for c in glist:
            wall_mem.setdefault(c, []).append(gname)

    print(f"\n  Camera positions (aligned frame: X=east, Y=north, Z=up, mm):")
    print(f"  {'Cam':>4}  {'X_mm':>9}  {'Y_mm':>9}  {'Z_mm':>9}  Groups")
    print(f"  {'-'*4}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*22}")
    for cid in sorted(cam_ids_s):
        if cid in failed:
            print(f"  {cid:>4}  {'— UNSOLVED —':>31}")
            continue
        C  = poses[cid]["C_aligned"]
        mg = ",".join(wall_mem.get(cid, ["-"]))
        print(f"  {cid:>4}  {float(C[0]):>9.0f}  {float(C[1]):>9.0f}  {float(C[2]):>9.0f}  {mg}")

    # Wall variance before/after
    grp_specs = [
        ("Wall A   X", WALL_A,  0, "C_zu_raw", "C_aligned"),
        ("Wall B   X", WALL_B,  0, "C_zu_raw", "C_aligned"),
        ("Wall C   Y", WALL_C,  1, "C_zu_raw", "C_aligned"),
        ("Wall D   Y", WALL_D,  1, "C_zu_raw", "C_aligned"),
        ("Ground   Z", GROUND,  2, "C_zu_raw", "C_aligned"),
        ("Ceiling  Z", CEILING, 2, "C_zu_raw", "C_aligned"),
        ("Mid-Z    Z", MID_Z,   2, "C_zu_raw", "C_aligned"),
    ]
    print(f"\n  Wall variance (std mm) — before vs after alignment:")
    print(f"  {'Group':>12}  {'Cameras':>32}  {'BEFORE':>8}  {'AFTER':>7}")
    print(f"  {'-'*12}  {'-'*32}  {'-'*8}  {'-'*7}")
    for label, grp, ax, k_b, k_a in grp_specs:
        gp = [c for c in grp if c in placed_ids]
        if len(gp) < 2:
            continue
        def _std(key):
            vals = [float(poses[c][key][ax]) for c in gp if key in poses[c]]
            return float(np.std(vals)) if len(vals) >= 2 else float('nan')
        print(f"  {label:>12}  {','.join(gp):>32}  {_std(k_b):>8.0f}  {_std(k_a):>7.0f}")

    # Height ordering
    print(f"\n  Height ordering (Z_mm, expect Ground < Mid < Ceiling < Top):")
    for gname, grp in [("Ground ",GROUND),("Mid    ",MID_Z),("Ceiling",CEILING),("Top    ",TOP)]:
        gp = [c for c in grp if c in placed_ids]
        if not gp:
            continue
        zv = [float(poses[c]["C_aligned"][2]) for c in gp]
        print(f"    {gname}: {[f'{z:.0f}' for z in zv]}  mean={np.mean(zv):.0f} mm")

    # Specific cam113 X report
    if "113" in placed_ids:
        print(f"\n  cam113 X position: {float(poses['113']['C_aligned'][0]):.0f} mm")
    else:
        print(f"\n  cam113: UNSOLVED")

    print(f"\n  Unsolved cameras: {failed if failed else 'none'}")


# ════════════════════════════════════════════════════════════════════════════
# Save + update config
# ════════════════════════════════════════════════════════════════════════════

def save_and_update(cam_ids_s, poses, cams, ref_cam, scale, cam_info, fps_map, failed):
    save = {"reference_camera":      np.array([ref_cam]),
            "reprojection_error_px": np.array([0.0]),
            "scale_factor":          np.array([scale])}
    for cid in cam_ids_s:
        if cid in poses:
            save[f"{cid}_R"] = poses[cid]["R"]
            save[f"{cid}_T"] = poses[cid]["T"]
        else:
            save[f"{cid}_R"] = np.eye(3, dtype=np.float64)
            save[f"{cid}_T"] = np.zeros((3,1), dtype=np.float64)
    np.savez(str(SCALED_NPZ), **save)
    print(f"\n  Saved → {SCALED_NPZ}")

    cfg = {}
    if CONFIG_JSON.exists():
        cfg = json.loads(CONFIG_JSON.read_text())

    cam_positions = {}
    for cid in cam_ids_s:
        if "C_aligned" in poses.get(cid, {}):
            C = poses[cid]["C_aligned"]
            cam_positions[cid] = [round(float(C[0]),1),
                                   round(float(C[1]),1),
                                   round(float(C[2]),1)]

    cam_s = sorted(cam_ids_s)
    w = cam_info[cam_s[0]]["w"]; h = cam_info[cam_s[0]]["h"]
    cfg["pricab_human_test"] = {
        "data_folder":           str(VIDEO_FOLDER),
        "cameras_found":         cam_s,
        "resolution":            f"{w}x{h}",
        "approach":              "YOLO11x-pose MST essmat+recoverPose wall-align",
        "reference_camera":      ref_cam,
        "scale_factor":          round(scale, 6),
        "camera_positions_mm":   cam_positions,
        "camera_fps":            fps_map,
        "unsolved_cameras":      failed,
        "alignment":             "Rz yaw (72-start) + Rx/Ry tilt + origin translate",
        "status":                "DONE",
        "timestamp":             datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    CONFIG_JSON.write_text(json.dumps(cfg, indent=2))
    print(f"  Updated → {CONFIG_JSON}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PriCaB_Rerun — MST essmat + wall alignment            ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Part 3 — FPS
    cam_info = read_fps_and_cam_info()
    fps_map  = {c: cam_info[c]["fps"] for c in cam_info}
    cam_ids  = sorted(cam_info.keys())

    # Load detections
    print(f"\n  Loading detections …")
    detections = _load_pose_files(cam_ids)

    # Load intrinsics
    _hdr(4, "INTRINSICS")
    cams = _load_intrinsics(cam_ids, cam_info)

    # Steps 1–5a: spanning tree + scale
    poses, ref_cam, scale, med_raw, failed, co_matrix = \
        run_extrinsics_spanning_tree(cam_ids, detections, cams)

    # Step 5b / Part 7: wall alignment
    cam_ids_s = sorted(cam_ids)
    poses, R_total, t_align = apply_wall_alignment(cam_ids_s, poses, failed, cams)

    # Save
    save_and_update(cam_ids_s, poses, cams, ref_cam, scale,
                    cam_info, fps_map, failed)

    # Step 6 / Part 8: validation table
    print_validation_table(cam_ids_s, poses, failed)

    import subprocess
    res = subprocess.run(["du","-sh",str(BASE_DIR)], capture_output=True, text=True)
    print(f"\n  Repo disk usage: {res.stdout.split()[0]}")
    print(f"\n{'═'*62}")
    print("DONE — awaiting review. No viewer rebuilt.")
    print('═'*62)


if __name__ == "__main__":
    main()
