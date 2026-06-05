#!/usr/bin/env python3
"""
Monkey GT Self-Calibration
==========================
Calibrate the 4 HomeCage cameras from human-labeled monkey 2D keypoints
(17 COCO joints per monkey per frame).  Uses the known camera positions and
the room floor as soft priors in bundle adjustment.

Pipeline
--------
  1. Load 4 per-cam CSVs for the chosen session.
  2. For each (frame, monkey_name, kp_idx) collect cross-camera correspondences.
  3. Pick the camera pair with the most simultaneous observations as anchor.
  4. recoverPose(E)  →  scale from the KNOWN inter-camera distance of the pair.
  5. Triangulate anchor-pair landmarks → solvePnP each remaining camera.
  6. Bundle adjustment with three residual blocks:
        (a) reprojection residuals (Huber)
        (b) Gaussian prior on each camera centre (sigma = SIGMA_CAM mm)
        (c) one-sided floor penalty on ankle keypoints (z < 0 → quadratic)
  7. Procrustes report against known positions, write JSON in the same schema
     as HomeCage_HumanCalib so the room viewer picks it up automatically.
"""
import csv, json, argparse
import numpy as np
import cv2
from pathlib import Path
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

# ─── Paths ────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).parent.parent
GT_DIR  = REPO / "_data/3d_validation/260402_HomeCage_702D_ElmJok/gt"
OUT_DIR = REPO / "HomeCage_SelfCalibration_Human/output"

# ─── Geometry ─────────────────────────────────────────────────────────────────
CAMERA_IDS = ["102", "108", "113", "117"]
KNOWN_POSITIONS = {
    "102": np.array([300.,  3260., 540.]),
    "108": np.array([1850., 0.,    2480.]),
    "113": np.array([50.,   0.,    550.]),
    "117": np.array([2080., 3070., 2550.]),
}
ROOM = {"x": 2240, "y": 3400, "z": 3260}

# Camera intrinsics (thin-lens)
SENSOR_W_MM, FOCAL_MM = 14.16, 8.0
IMG_W, IMG_H          = 2048, 1496

# ─── Keypoints ────────────────────────────────────────────────────────────────
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
ANKLE_IDX = [15, 16]
KP_CONF   = 0.0   # GT labels: presence == valid

# ─── BA priors ────────────────────────────────────────────────────────────────
FLOOR_WEIGHT    = 50.0    # weight for ankle z<0 penalty
SIGMA_REPROJ_PX = 4.0     # noise model for reprojection residuals
MAX_LANDMARKS   = 1500    # cap BA landmarks (randomly sampled if more)
# Camera centres are FIXED at KNOWN_POSITIONS — only rotations & 3D points are free.


def K_thinlens(w_px, h_px):
    fx = FOCAL_MM * w_px / SENSOR_W_MM
    return np.array([[fx, 0, w_px / 2.0],
                     [0, fx, h_px / 2.0],
                     [0,  0,         1.0]])


# ─── 1. Load CSV ──────────────────────────────────────────────────────────────
def load_session(session):
    """Return {cam: {(session, frame, monkey): [17 × (x,y,valid)]}}.
    Keys are tagged with the session id so frames from different sessions
    don't collide."""
    data = {}
    for cam in CAMERA_IDS:
        path = GT_DIR / f"gt_{session}_cam{cam}.csv"
        if not path.exists():
            continue
        cam_dict = {}
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                frame  = int(row["frame_number"])
                monkey = row["monkey_name"]
                kps = []
                ok = False
                for name in KP_NAMES:
                    sx, sy = row.get(f"{name}_x", ""), row.get(f"{name}_y", "")
                    if sx.strip() and sy.strip():
                        kps.append((float(sx), float(sy), 1.0))
                        ok = True
                    else:
                        kps.append((0.0, 0.0, 0.0))
                if ok:
                    cam_dict[(session, frame, monkey)] = kps
        data[cam] = cam_dict
        print(f"  {session} cam{cam}: {len(cam_dict)} labeled detections")
    return data


def load_sessions(sessions):
    """Merge multiple sessions into a single {cam: {key: kps}} dict."""
    merged = {}
    for sess in sessions:
        d = load_session(sess)
        for cam, cam_dict in d.items():
            merged.setdefault(cam, {}).update(cam_dict)
    print(f"\n  Merged: " + ", ".join(
        f"cam{c}={len(merged[c])}" for c in CAMERA_IDS if c in merged))
    return merged


# ─── 2. Helpers ───────────────────────────────────────────────────────────────
def dlt(Ps, pts):
    rows = []
    for P, p in zip(Ps, pts):
        rows += [p[0]*P[2] - P[0], p[1]*P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def cam_centre(R, t):
    return (-R.T @ t.flatten())


def project(K, R, t, X):
    p = K @ (R @ X + t.flatten())
    return p[:2] / p[2]


def procrustes(src, dst):
    sc, dc = src.mean(0), dst.mean(0)
    A = (dst - dc).T @ (src - sc)
    U, _, Vt = np.linalg.svd(A)
    D = np.diag([1., 1., np.linalg.det(U @ Vt)])
    R = U @ D @ Vt
    t = dc - R @ sc
    return R, t


# ─── 3. Anchor pair selection ─────────────────────────────────────────────────
def best_pair(data):
    """Return (cam_a, cam_b, list of (key, kp_idx) common observations)."""
    best, best_n = None, 0
    for i, ca in enumerate(CAMERA_IDS):
        if ca not in data: continue
        for cb in CAMERA_IDS[i+1:]:
            if cb not in data: continue
            common_keys = set(data[ca]) & set(data[cb])
            n = 0
            for k in common_keys:
                kpa, kpb = data[ca][k], data[cb][k]
                for j in range(17):
                    if kpa[j][2] > 0 and kpb[j][2] > 0:
                        n += 1
            if n > best_n:
                best_n, best = n, (ca, cb)
    return best, best_n


# ─── 4. Initial pose ──────────────────────────────────────────────────────────
def init_two_view(data, ca, cb, K):
    pts_a, pts_b = [], []
    for k in set(data[ca]) & set(data[cb]):
        kpa, kpb = data[ca][k], data[cb][k]
        for j in range(17):
            if kpa[j][2] > 0 and kpb[j][2] > 0:
                pts_a.append(kpa[j][:2])
                pts_b.append(kpb[j][:2])
    pts_a = np.array(pts_a, np.float64)
    pts_b = np.array(pts_b, np.float64)
    E, _ = cv2.findEssentialMat(pts_a, pts_b, K,
                                method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, K)
    return R, t  # unit-scale


def metric_scale(ca, cb):
    return float(np.linalg.norm(KNOWN_POSITIONS[ca] - KNOWN_POSITIONS[cb]))


# ─── 5. Add remaining cameras via PnP ─────────────────────────────────────────
def triangulate_landmarks(data, placed, Ks, Rs, Ts):
    """Return {(key, kp_idx): X3d}."""
    Ps = {c: Ks[c] @ np.hstack([Rs[c], Ts[c].reshape(3,1)]) for c in placed}
    landmarks = {}
    keys = set()
    for c in placed:
        keys |= set(data[c])
    for k in keys:
        for j in range(17):
            obs_P, obs_p = [], []
            for c in placed:
                if k not in data[c]:
                    continue
                kp = data[c][k]
                if kp[j][2] > 0:
                    obs_P.append(Ps[c])
                    obs_p.append(kp[j][:2])
            if len(obs_P) >= 2:
                landmarks[(k, j)] = dlt(obs_P, obs_p)
    return landmarks


def pnp_camera(data, cam, K, landmarks):
    obj_pts, img_pts = [], []
    for (key, j), X in landmarks.items():
        if key in data[cam] and data[cam][key][j][2] > 0:
            obj_pts.append(X)
            img_pts.append(data[cam][key][j][:2])
    if len(obj_pts) < 6:
        return None
    obj = np.array(obj_pts, np.float64)
    img = np.array(img_pts, np.float64)
    ok, rvec, tvec, _ = cv2.solvePnPRansac(obj, img, K, None, reprojectionError=8.0)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec


# ─── 6. Bundle adjustment with priors ─────────────────────────────────────────
def bundle_adjust(data, placed, Ks, Rs, Ts):
    landmarks = triangulate_landmarks(data, placed, Ks, Rs, Ts)
    print(f"  Initial landmarks: {len(landmarks)}", flush=True)

    if len(landmarks) > MAX_LANDMARKS:
        rng = np.random.default_rng(0)
        keys = list(landmarks.keys())
        idx  = rng.choice(len(keys), MAX_LANDMARKS, replace=False)
        landmarks = {keys[i]: landmarks[keys[i]] for i in idx}
        print(f"  Sampled down to {len(landmarks)} for BA", flush=True)

    lm_keys = list(landmarks.keys())
    lm_idx  = {k: i for i, k in enumerate(lm_keys)}
    cam_list = list(placed)
    cam_idx  = {c: i for i, c in enumerate(cam_list)}

    # Camera centres FIXED at known positions; only rotations free.
    # Pack: [rvec(3) per cam, X(3) per landmark]
    n_cam = len(cam_list)
    n_lm  = len(lm_keys)
    C_fix = np.array([KNOWN_POSITIONS[c] for c in cam_list])  # (n_cam, 3)

    x0 = np.zeros(n_cam * 3 + n_lm * 3)
    for c in cam_list:
        i = cam_idx[c]
        rvec, _ = cv2.Rodrigues(Rs[c])
        x0[i*3 : i*3+3] = rvec.flatten()
    for k in lm_keys:
        x0[n_cam*3 + lm_idx[k]*3 : n_cam*3 + lm_idx[k]*3 + 3] = landmarks[k]

    # Pre-build observation arrays
    obs_ci, obs_li, obs_uv = [], [], []
    for c in cam_list:
        for k in lm_keys:
            key, j = k
            if key in data[c] and data[c][key][j][2] > 0:
                obs_ci.append(cam_idx[c])
                obs_li.append(lm_idx[k])
                obs_uv.append((data[c][key][j][0], data[c][key][j][1]))
    obs_ci = np.asarray(obs_ci, np.int32)
    obs_li = np.asarray(obs_li, np.int32)
    obs_uv = np.asarray(obs_uv, np.float64)
    print(f"  Observations: {len(obs_ci)}")

    Ks_arr = np.stack([Ks[c] for c in cam_list])  # (n_cam,3,3)

    ankle_li = np.array([li for li, k in enumerate(lm_keys) if k[1] in ANKLE_IDX],
                        dtype=np.int32)

    def residuals(x):
        # rotations and derived translations (t = -R @ C_fix)
        Rs_a = np.empty((n_cam, 3, 3))
        for ci in range(n_cam):
            R, _ = cv2.Rodrigues(x[ci*3 : ci*3+3])
            Rs_a[ci] = R
        ts_a = -np.einsum('nij,nj->ni', Rs_a, C_fix)            # (n_cam,3)
        Xs = x[n_cam*3:].reshape(n_lm, 3)

        R_obs = Rs_a[obs_ci]
        t_obs = ts_a[obs_ci]
        K_obs = Ks_arr[obs_ci]
        X_obs = Xs[obs_li]
        Xc    = np.einsum('nij,nj->ni', R_obs, X_obs) + t_obs
        p     = np.einsum('nij,nj->ni', K_obs, Xc)
        z = p[:, 2]
        bad = z <= 0
        z_safe = np.where(bad, 1.0, z)
        uv = p[:, :2] / z_safe[:, None]
        diff = (uv - obs_uv) / SIGMA_REPROJ_PX
        diff[bad] = 1e3
        res_reproj = diff.reshape(-1)

        if ankle_li.size:
            zk = Xs[ankle_li, 2]
            res_floor = FLOOR_WEIGHT * np.where(zk < 0, -zk, 0.0) / 100.0
        else:
            res_floor = np.array([])

        return np.concatenate([res_reproj, res_floor])

    # Jacobian sparsity
    n_obs = len(obs_ci)
    n_floor = int(ankle_li.size)
    n_res = 2*n_obs + n_floor
    n_par = n_cam*3 + n_lm*3
    sp = lil_matrix((n_res, n_par), dtype=int)
    for i in range(n_obs):
        ci = obs_ci[i]; li = obs_li[i]
        sp[2*i:2*i+2, ci*3:ci*3+3] = 1
        sp[2*i:2*i+2, n_cam*3 + li*3 : n_cam*3 + li*3 + 3] = 1
    for fi, li in enumerate(ankle_li):
        sp[2*n_obs + fi, n_cam*3 + li*3 + 2] = 1

    print("  Running bundle adjustment…", flush=True)
    res0 = np.array(residuals(x0))
    print(f"    initial cost: {0.5*np.sum(res0**2):.1f}  "
          f"(median |r|={np.median(np.abs(res0)):.3f})", flush=True)

    result = least_squares(residuals, x0, method="trf", loss="soft_l1",
                           f_scale=2.0, max_nfev=300, verbose=2,
                           jac_sparsity=sp, x_scale='jac')
    x = result.x
    res1 = np.array(residuals(x))
    print(f"    final   cost: {0.5*np.sum(res1**2):.1f}  "
          f"(median |r|={np.median(np.abs(res1)):.3f})")

    Rs_out, Ts_out = {}, {}
    for c in cam_list:
        i = cam_idx[c]
        rvec = x[i*3 : i*3+3]
        R, _ = cv2.Rodrigues(rvec)
        Rs_out[c] = R
        Ts_out[c] = -R @ KNOWN_POSITIONS[c]

    # Reprojection error in pixels
    px_errs = []
    Xs_final = x[n_cam*3:].reshape(n_lm, 3)
    for i in range(len(obs_ci)):
        c = cam_list[obs_ci[i]]
        X = Xs_final[obs_li[i]]
        p = project(Ks[c], Rs_out[c], Ts_out[c], X)
        px_errs.append(np.linalg.norm(p - obs_uv[i]))
    print(f"    reprojection: median={np.median(px_errs):.2f}px  "
          f"mean={np.mean(px_errs):.2f}px")
    return Rs_out, Ts_out, float(np.median(px_errs))


# ─── 7. Main ──────────────────────────────────────────────────────────────────
def look_at(C, target, up=np.array([0, 0, 1.0])):
    """Return world→camera rotation R such that camera at C looks at target."""
    fwd = target - C
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    new_up = np.cross(right, fwd)
    # OpenCV convention: cam X right, Y down, Z forward
    R = np.stack([right, -new_up, fwd])  # rows
    return R


def calibrate(sessions):
    if isinstance(sessions, str):
        sessions = [sessions]
    label = sessions[0] if len(sessions) == 1 else "all"
    print(f"\n── Monkey GT Self-Calibration · sessions {','.join(sessions)} ──")
    data = load_sessions(sessions) if len(sessions) > 1 else load_session(sessions[0])
    if len(data) < 2:
        print("  Need ≥ 2 cameras")
        return

    Ks = {c: K_thinlens(IMG_W, IMG_H) for c in CAMERA_IDS}
    placed = set(data.keys())

    # Initialize each camera by pointing it at the room centre
    room_centre = np.array([ROOM["x"]/2, ROOM["y"]/2, ROOM["z"]/4])
    Rs, Ts = {}, {}
    for c in placed:
        R = look_at(KNOWN_POSITIONS[c], room_centre)
        Rs[c] = R
        Ts[c] = -R @ KNOWN_POSITIONS[c]
    print(f"  Initialized {len(placed)} cameras (look-at room centre)")

    Rs, Ts, reproj_med = bundle_adjust(data, placed, Ks, Rs, Ts)

    # ── Procrustes to known positions for reporting ──
    common = sorted(placed)
    src = np.array([cam_centre(Rs[c], Ts[c]) for c in common])
    tgt = np.array([KNOWN_POSITIONS[c]       for c in common])
    R_align, t_align = procrustes(src, tgt)
    aligned = {c: R_align @ src[i] + t_align for i, c in enumerate(common)}
    errors  = {c: float(np.linalg.norm(aligned[c] - KNOWN_POSITIONS[c])) for c in common}
    mean_err = float(np.mean(list(errors.values())))
    print(f"\n  Position errors after Procrustes:")
    for c in CAMERA_IDS:
        if c in errors:
            print(f"    cam{c}: {errors[c]:.0f} mm")
    print(f"  Mean: {mean_err:.0f} mm")

    # ── Save in same schema as HomeCage_HumanCalib ──
    result = {
        "sessions":         sessions,
        "method":           "monkey_gt",
        "label":            f"Monkey GT {label}",
        "init_method":      "look_at_room_centre",
        "n_cameras":        len(placed),
        "reproj_after_ba":  reproj_med,
        "floor_weight":     FLOOR_WEIGHT,
        "positions_fixed":  True,
        "cameras":          {},
    }
    for c in CAMERA_IDS:
        entry = {
            "known_mm": KNOWN_POSITIONS[c].tolist(),
            "placed":   c in placed,
        }
        if c in aligned:
            entry["reconstructed_aligned_mm"] = [round(v, 1) for v in aligned[c].tolist()]
            entry["error_mm"] = round(errors[c], 1)
        if c in placed:
            R_w = Rs[c] @ R_align.T
            T_w = Ts[c] - Rs[c] @ R_align.T @ t_align
            entry["R_world"] = [[round(v, 8) for v in row] for row in R_w.tolist()]
            entry["T_world"] = [round(v, 4) for v in T_w.tolist()]
            entry["K"]       = [[round(v, 4) for v in row] for row in Ks[c].tolist()]
        result["cameras"][c] = entry
    result["mean_error_mm"] = round(mean_err, 1)

    out_path = OUT_DIR / f"monkey_gt_{label}_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default="250708",
                    help="comma-separated session dates (default: 250708)")
    args = ap.parse_args()
    sessions = [s.strip() for s in args.sessions.split(",") if s.strip()]
    calibrate(sessions)
