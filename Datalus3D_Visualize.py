#!/usr/bin/env python3
"""
Datalus 3D Visualizer

Two modes:

  1. YOLO mode (default) — runs YOLOv8 pose on images/videos:
       python Datalus3D_Visualize.py img_102.png img_108.png img_113.png img_117.png
       python Datalus3D_Visualize.py vid_102.mp4 vid_108.mp4 vid_113.mp4 vid_117.mp4 --t 60
       python Datalus3D_Visualize.py vid_102.mp4 ... --loop --step 1

  2. Results mode — uses pre-computed 2D txt files (no YOLO needed):
       python Datalus3D_Visualize.py --from-results \\
           result_102.txt result_108.txt result_113.txt result_117.txt
       python Datalus3D_Visualize.py --from-results result_*.txt --monkey Elm
       python Datalus3D_Visualize.py --from-results result_*.txt --loop
       python Datalus3D_Visualize.py --from-results result_*.txt --loop \\
           --videos vid_102.mp4 vid_108.mp4 vid_113.mp4 vid_117.mp4

     The script automatically finds the frame where the selected monkey is
     visible in the most cameras with highest confidence.

Requires: pip install ultralytics opencv-python matplotlib
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path

YAML_DIR   = "/Users/acalapai/PycharmProjects/Datalus/DatalusCalibration/yaml"
YOLO_MODEL = "/Users/acalapai/PycharmProjects/Datalus/yolov8n-pose.pt"
CAMERA_IDS = ["102", "108", "113", "117"]

CONF_THRESH = 0.3   # minimum keypoint confidence to use for triangulation

# COCO 17-keypoint skeleton
KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]


# ── Calibration ───────────────────────────────────────────────────────────────

def read_opencv_matrix(yaml_node) -> np.ndarray:
    return yaml_node.mat()


def load_calibration(yaml_dir: str, camera_ids: list) -> dict:
    """Returns dict cam_id -> {K, dist, R, T}."""
    cams = {}
    for cam_id in camera_ids:
        p = Path(yaml_dir) / f"{cam_id}.yaml"
        if not p.exists():
            print(f"[ERROR] YAML not found: {p}")
            print("  Run DatulusCalib_Step4_WriteYAMLs.py first.")
            sys.exit(1)
        fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
        K    = read_opencv_matrix(fs.getNode("intrinsicMatrix")).T
        dist = read_opencv_matrix(fs.getNode("distortionCoefficients")).ravel()
        R    = read_opencv_matrix(fs.getNode("R")).T
        T    = read_opencv_matrix(fs.getNode("T")).reshape(3, 1)
        fs.release()
        cams[cam_id] = dict(K=K, dist=dist, R=R, T=T)
    return cams


# ── Results file parsing ──────────────────────────────────────────────────────

def parse_results_file(path: str) -> dict:
    """
    Parse an ABT 2D result txt file.
    Returns dict: frame_num (int) -> {monkey_id (str): kps (17,3) array}

    File format (space-separated data, comma-separated header):
      frame_number monkey_ID bbox_x1 bbox_y1 bbox_x2 bbox_y2 bbox_conf
      nose_x nose_y nose_conf  left_eye_x ... (17 keypoints × 3 values)
    """
    results = {}
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or i == 0:   # skip header
                continue
            cols = line.split()
            if len(cols) < 7 + 17 * 3:
                continue
            frame = int(cols[0])
            monkey = cols[1]
            kps = np.array([float(v) for v in cols[7:]], dtype=np.float32)
            kps = kps[:17 * 3].reshape(17, 3)   # (17, 3): x, y, conf

            if frame not in results:
                results[frame] = {}
            results[frame][monkey] = kps
    return results


def find_best_frames(results_per_cam: dict, monkey= None,
                     min_cams: int = 2) -> list:
    """
    Find frames where the monkey is detected in >= min_cams cameras.

    results_per_cam: {cam_id: {frame: {monkey: kps}}}
    Returns list of dicts sorted by (n_cams DESC, mean_conf DESC):
      {frame, monkey, n_cams, mean_conf, detections: {cam_id: kps}}
    """
    # Collect all frames and monkeys across cameras
    all_frames = set()
    all_monkeys = set()
    for cam_data in results_per_cam.values():
        for frame, monkeys in cam_data.items():
            all_frames.add(frame)
            all_monkeys.update(monkeys.keys())

    if monkey is None:
        candidates = sorted(all_monkeys)
    else:
        candidates = [monkey]

    ranked = []
    for frame in sorted(all_frames):
        for m in candidates:
            dets = {}
            for cam_id, cam_data in results_per_cam.items():
                if frame in cam_data and m in cam_data[frame]:
                    dets[cam_id] = cam_data[frame][m]
            if len(dets) < min_cams:
                continue
            # Mean confidence across all detected keypoints in all cameras
            conf = np.mean([kps[:, 2].mean() for kps in dets.values()])
            ranked.append(dict(frame=frame, monkey=m, n_cams=len(dets),
                               mean_conf=float(conf), detections=dets))

    ranked.sort(key=lambda x: (-x["n_cams"], -x["mean_conf"]))
    return ranked


# ── Frame loading ──────────────────────────────────────────────────────────────

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".MP4", ".MOV"}


def load_frame_at(path: str, frame_num: int):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def load_frame(path: str, t_sec: float) -> np.ndarray:
    p = Path(path)
    if p.suffix.lower() in {e.lower() for e in VIDEO_EXTS}:
        cap = cv2.VideoCapture(str(p))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t_sec * fps))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"[ERROR] Could not read frame at t={t_sec}s from {path}")
            sys.exit(1)
        return frame
    else:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[ERROR] Could not read image: {path}")
            sys.exit(1)
        return img


def video_frame_generator(path: str, step_sec: float):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(step_sec * fps))
    idx  = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            yield idx / fps, frame
        idx += 1
    cap.release()


# ── YOLO detection ────────────────────────────────────────────────────────────

def detect_keypoints(model, frame: np.ndarray):
    results = model(frame, verbose=False)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints.data
    if kps.shape[0] == 0:
        return None
    best = int(kps[:, :, 2].mean(dim=1).argmax())
    return kps[best].cpu().numpy()


# ── Triangulation ─────────────────────────────────────────────────────────────

def triangulate_kp(kp_idx: int, detections: dict, cams: dict):
    # Collect normalised rays per camera
    cam_rays = []   # (cam_id, xu, yu, P)
    for cam_id, kps in detections.items():
        if kps is None:
            continue
        x, y, conf = kps[kp_idx]
        if conf < CONF_THRESH:
            continue
        K, dist = cams[cam_id]["K"], cams[cam_id]["dist"]
        pt_ud = cv2.undistortPoints(
            np.array([[[x, y]]], dtype=np.float32), K, dist)[0, 0]
        R, T = cams[cam_id]["R"], cams[cam_id]["T"]
        P = np.hstack([R, T])
        cam_rays.append((cam_id, pt_ud[0], pt_ud[1], P))

    if len(cam_rays) < 2:
        return None

    def _dlt(rays):
        rows = []
        for _, xu, yu, P in rays:
            rows.append(xu * P[2] - P[0])
            rows.append(yu * P[2] - P[1])
        A = np.array(rows)
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            return None
        return X[:3] / X[3]

    X3 = _dlt(cam_rays)
    if X3 is None:
        return None

    # Drop cameras where depth Z ≤ 0 (point behind camera) and redo
    good = []
    for item in cam_rays:
        cam_id, xu, yu, P = item
        R = cams[cam_id]["R"]; T = cams[cam_id]["T"]
        Z = (R @ X3 + T.ravel())[2]
        if Z > 1e-4:
            good.append(item)

    if len(good) < 2:
        return None
    if len(good) < len(cam_rays):
        X3 = _dlt(good)

    return X3


def triangulate_all(detections: dict, cams: dict) -> np.ndarray:
    pts3d = np.full((17, 3), np.nan)
    for k in range(17):
        p = triangulate_kp(k, detections, cams)
        if p is not None:
            pts3d[k] = p
    return pts3d


def reprojection_error(pts3d: np.ndarray, detections: dict, cams: dict) -> float:
    errs = []
    for cam_id, kps in detections.items():
        if kps is None:
            continue
        K, dist = cams[cam_id]["K"], cams[cam_id]["dist"]
        R, T    = cams[cam_id]["R"], cams[cam_id]["T"]
        for k in range(17):
            if np.any(np.isnan(pts3d[k])):
                continue
            x_obs, y_obs, conf = kps[k]
            if conf < CONF_THRESH:
                continue
            proj, _ = cv2.projectPoints(
                pts3d[k].reshape(1, 3), R, T, K, dist)
            xp, yp = proj[0, 0]
            errs.append(np.sqrt((xp - x_obs)**2 + (yp - y_obs)**2))
    return float(np.mean(errs)) if errs else float("nan")


# ── Visualization ─────────────────────────────────────────────────────────────

def draw_2d(frame: np.ndarray, kps) -> np.ndarray:
    out = frame.copy()
    if kps is None:
        return out
    for k in range(17):
        x, y, conf = kps[k]
        if conf >= CONF_THRESH:
            cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
    for a, b in SKELETON:
        if kps[a, 2] >= CONF_THRESH and kps[b, 2] >= CONF_THRESH:
            cv2.line(out, (int(kps[a,0]), int(kps[a,1])),
                         (int(kps[b,0]), int(kps[b,1])), (0, 200, 255), 2)
    return out


def show_2d_grid(frames: dict, detections: dict):
    annotated = []
    for cam_id in CAMERA_IDS:
        frame = frames.get(cam_id)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ann = draw_2d(frame, detections.get(cam_id))
        h, w = ann.shape[:2]
        scale = 640 / max(h, w)
        ann = cv2.resize(ann, (int(w * scale), int(h * scale)))
        cv2.putText(ann, f"cam {cam_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        annotated.append(ann)

    top = np.hstack(annotated[:2])
    bot = np.hstack(annotated[2:])
    if top.shape[1] != bot.shape[1]:
        w = max(top.shape[1], bot.shape[1])
        top = cv2.copyMakeBorder(top, 0, 0, 0, w - top.shape[1], cv2.BORDER_CONSTANT)
        bot = cv2.copyMakeBorder(bot, 0, 0, 0, w - bot.shape[1], cv2.BORDER_CONSTANT)
    cv2.imshow("2D detections", np.vstack([top, bot]))
    cv2.waitKey(1)


def plot_3d(ax, pts3d: np.ndarray, cams: dict, title: str):
    ax.cla()
    valid = ~np.any(np.isnan(pts3d), axis=1)

    if valid.any():
        ax.scatter(pts3d[valid, 0], pts3d[valid, 1], pts3d[valid, 2],
                   c="lime", s=60, zorder=5)
        for k in range(17):
            if valid[k]:
                ax.text(pts3d[k,0], pts3d[k,1], pts3d[k,2],
                        f" {KP_NAMES[k]}", fontsize=6, color="white")
        for a, b in SKELETON:
            if valid[a] and valid[b]:
                ax.plot([pts3d[a,0], pts3d[b,0]],
                        [pts3d[a,1], pts3d[b,1]],
                        [pts3d[a,2], pts3d[b,2]], "c-", lw=2)

    for cam_id, c in cams.items():
        C = (-c["R"].T @ c["T"]).ravel()
        ax.scatter(*C, marker="^", s=120, c="red", zorder=6)
        ax.text(C[0], C[1], C[2], f" cam{cam_id}", fontsize=8, color="red")

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.set_title(title, color="white")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs=4,
                   metavar="PATH",
                   help="4 files: images/videos (YOLO mode) or result txts (--from-results)")
    p.add_argument("--from-results", action="store_true",
                   help="Use pre-computed 2D result txt files instead of running YOLO")
    p.add_argument("--monkey", type=str, default=None,
                   help="Which monkey to track (e.g. Elm or Jok). Default: auto best.")
    p.add_argument("--videos", nargs=4, default=None,
                   metavar="VID",
                   help="Optional: 4 RAW video files for 2D image overlay in --from-results mode")
    p.add_argument("--loop", action="store_true",
                   help="Loop through all frames (both modes)")
    p.add_argument("--step", type=float, default=1.0,
                   help="Step between frames in seconds (YOLO loop mode, default: 1)")
    p.add_argument("--t", type=float, default=10.0,
                   help="Timestamp to extract for single-frame YOLO mode (default: 10)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading calibration...")
    cams = load_calibration(YAML_DIR, CAMERA_IDS)
    print(f"  Loaded {len(cams)} cameras")

    import matplotlib
    _display = os.environ.get("DISPLAY", "") or os.environ.get("TERM_PROGRAM", "")
    _backend = "MacOSX" if sys.platform == "darwin" and _display else "Agg"
    matplotlib.use(_backend)
    import matplotlib.pyplot as plt

    # ── Results mode ──────────────────────────────────────────────────────────
    if args.from_results:
        path_map = dict(zip(CAMERA_IDS, args.paths))
        vid_map  = dict(zip(CAMERA_IDS, args.videos)) if args.videos else {}

        print("Parsing 2D result files...")
        results_per_cam = {}
        for cam_id, path in path_map.items():
            results_per_cam[cam_id] = parse_results_file(path)
            n = sum(len(m) for m in results_per_cam[cam_id].values())
            print(f"  cam {cam_id}: {len(results_per_cam[cam_id])} frames, "
                  f"{n} detections")

        ranked = find_best_frames(results_per_cam, args.monkey, min_cams=2)
        if not ranked:
            print("[ERROR] No frames found with 2+ cameras detecting the same monkey.")
            sys.exit(1)

        # Print summary of what's available
        monkeys = sorted({r["monkey"] for r in ranked})
        for m in monkeys:
            best = next(r for r in ranked if r["monkey"] == m)
            total = sum(1 for r in ranked if r["monkey"] == m)
            print(f"  {m}: {total} usable frames, "
                  f"best is frame {best['frame']} "
                  f"({best['n_cams']} cams, conf={best['mean_conf']:.2f})")

        fig = plt.figure(figsize=(10, 8))
        ax  = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("#1a1a2e")

        if args.loop:
            plt.ion()
            frames_to_show = [r for r in ranked
                              if args.monkey is None or r["monkey"] == args.monkey]
            # Sort chronologically for loop
            frames_to_show.sort(key=lambda x: x["frame"])
            print(f"\nLooping {len(frames_to_show)} frames. Ctrl+C to stop.")

            try:
                for entry in frames_to_show:
                    detections = {cam_id: entry["detections"].get(cam_id)
                                  for cam_id in CAMERA_IDS}
                    pts3d = triangulate_all(detections, cams)
                    err   = reprojection_error(pts3d, detections, cams)
                    valid = int((~np.any(np.isnan(pts3d), axis=1)).sum())

                    title = (f"frame {entry['frame']}  |  {entry['monkey']}  |  "
                             f"{entry['n_cams']} cams  |  {valid}/17 kp  |  "
                             f"reproj={err:.1f}px")
                    print(f"  {title}")
                    plot_3d(ax, pts3d, cams, title)

                    if vid_map:
                        imgs = {cam_id: load_frame_at(vid_map[cam_id], entry["frame"])
                                for cam_id in CAMERA_IDS if cam_id in vid_map}
                        show_2d_grid(imgs, detections)

                    plt.pause(0.05)

            except KeyboardInterrupt:
                print("\nStopped.")

        else:
            # Single best frame
            if args.monkey:
                entry = next((r for r in ranked if r["monkey"] == args.monkey), None)
                if entry is None:
                    print(f"[ERROR] Monkey '{args.monkey}' not found in results.")
                    sys.exit(1)
            else:
                entry = ranked[0]

            print(f"\nBest frame: {entry['frame']}  monkey={entry['monkey']}  "
                  f"cams={entry['n_cams']}  conf={entry['mean_conf']:.2f}")

            detections = {cam_id: entry["detections"].get(cam_id)
                          for cam_id in CAMERA_IDS}
            pts3d = triangulate_all(detections, cams)
            err   = reprojection_error(pts3d, detections, cams)
            valid = int((~np.any(np.isnan(pts3d), axis=1)).sum())
            print(f"  {valid}/17 keypoints triangulated  reproj={err:.2f}px")

            title = (f"frame {entry['frame']}  |  {entry['monkey']}  |  "
                     f"{entry['n_cams']} cams  |  {valid}/17 kp  |  "
                     f"reproj={err:.1f}px")
            plot_3d(ax, pts3d, cams, title)

            if vid_map:
                imgs = {cam_id: load_frame_at(vid_map[cam_id], entry["frame"])
                        for cam_id in CAMERA_IDS if cam_id in vid_map}
                show_2d_grid(imgs, detections)

            out_png = Path(__file__).parent / "visualization_output.png"
            plt.savefig(str(out_png), dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"\n  Figure saved → {out_png}")
            try:
                plt.show()
            except Exception:
                pass

        cv2.destroyAllWindows()
        plt.close("all")
        return

    # ── YOLO mode ─────────────────────────────────────────────────────────────
    print("Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO(YOLO_MODEL)

    path_map = dict(zip(CAMERA_IDS, args.paths))
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#1a1a2e")

    if args.loop:
        gens = {cam_id: video_frame_generator(path_map[cam_id], args.step)
                for cam_id in CAMERA_IDS}
        plt.ion()
        try:
            while True:
                frames, t_sec = {}, None
                for cam_id, gen in gens.items():
                    try:
                        t, frame = next(gen)
                        frames[cam_id] = frame
                        if t_sec is None:
                            t_sec = t
                    except StopIteration:
                        print("End of video.")
                        return

                detections = {cam_id: detect_keypoints(model, frames[cam_id])
                              for cam_id in CAMERA_IDS}
                n_det = sum(1 for v in detections.values() if v is not None)
                print(f"  t={t_sec:.1f}s  detected in {n_det}/4 cameras", end="")

                if n_det >= 2:
                    pts3d = triangulate_all(detections, cams)
                    err   = reprojection_error(pts3d, detections, cams)
                    valid = int((~np.any(np.isnan(pts3d), axis=1)).sum())
                    print(f"  →  {valid} kp  reproj={err:.1f}px")
                    title = f"t={t_sec:.1f}s  |  {n_det}/4 cams  |  {valid}/17 kp  |  reproj={err:.1f}px"
                    plot_3d(ax, pts3d, cams, title)
                    show_2d_grid(frames, detections)
                    plt.pause(0.001)
                else:
                    print("  →  skipped")

        except KeyboardInterrupt:
            print("\nStopped.")

    else:
        print(f"\nExtracting frames at t={args.t}s...")
        frames = {cam_id: load_frame(path_map[cam_id], args.t)
                  for cam_id in CAMERA_IDS}

        print("Running YOLO pose detection...")
        detections = {}
        for cam_id in CAMERA_IDS:
            kps = detect_keypoints(model, frames[cam_id])
            detections[cam_id] = kps
            status = f"{int((kps[:,2] >= CONF_THRESH).sum())} kp" if kps is not None else "no detection"
            print(f"  cam {cam_id}: {status}")

        n_det = sum(1 for v in detections.values() if v is not None)
        if n_det < 2:
            print("[ERROR] Need detections in at least 2 cameras to triangulate.")
            sys.exit(1)

        pts3d = triangulate_all(detections, cams)
        err   = reprojection_error(pts3d, detections, cams)
        valid = int((~np.any(np.isnan(pts3d), axis=1)).sum())
        print(f"  {valid}/17 keypoints triangulated  reproj={err:.2f}px")

        title = f"t={args.t:.1f}s  |  {n_det}/4 cams  |  {valid}/17 kp  |  reproj={err:.1f}px"
        plot_3d(ax, pts3d, cams, title)
        show_2d_grid(frames, detections)

        print("\nPress Enter to close.")
        input()

    cv2.destroyAllWindows()
    plt.close("all")


if __name__ == "__main__":
    main()
