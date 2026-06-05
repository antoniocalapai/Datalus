#!/usr/bin/env python3
"""Triangulate per-frame 17 COCO keypoints from the 4 calibrated cameras.

Reads:
  HomeCage_SelfCalibration_Human/output/calibration_result.json   (R, T, K per cam)
  HomeCage_SelfCalibration_Human/output/pose_<cam>_250708.txt     (2D detections)

Writes:
  HomeCage_SelfCalibration_Human/output/human_track.json
    {frames: [{f, kps:[[x,y,z]|null × 17]}]}
"""
import json
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent / "HomeCage_SelfCalibration_Human" / "output"
CAMS = ["102", "108", "113", "117"]
SESSION = "250708"
KP_CONF = 0.50
MIN_CAMS = 2


def load_pose(path):
    """Return {frame: 17x3 ndarray (x,y,conf)}."""
    out = {}
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    i = 0
    while i < len(lines):
        head = lines[i].split()
        kp_line = lines[i + 1].split()
        i += 2
        frame = int(head[0])
        kps = np.array(kp_line, dtype=float).reshape(17, 3)
        out[frame] = kps
    return out


def dlt(Ps, pts):
    rows = []
    for P, p in zip(Ps, pts):
        rows += [p[0] * P[2] - P[0], p[1] * P[2] - P[1]]
    _, _, Vt = np.linalg.svd(np.stack(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def main():
    with open(OUT / "calibration_result.json") as f:
        calib = json.load(f)

    Ps = {}
    for cam in CAMS:
        info = calib["cameras"][cam]
        K = np.array(info["K"])
        R = np.array(info["R_world"])
        T = np.array(info["T_world"]).reshape(3, 1)
        Ps[cam] = K @ np.hstack([R, T])

    poses = {cam: load_pose(OUT / f"pose_{cam}_{SESSION}.txt") for cam in CAMS}
    all_frames = sorted(set().union(*(p.keys() for p in poses.values())))

    frames_out = []
    for f in all_frames:
        kps3d = [None] * 17
        for k in range(17):
            obs_P, obs_p = [], []
            for cam in CAMS:
                if f not in poses[cam]:
                    continue
                kp = poses[cam][f][k]
                if kp[2] >= KP_CONF:
                    obs_P.append(Ps[cam])
                    obs_p.append(kp[:2])
            if len(obs_P) >= MIN_CAMS:
                X = dlt(obs_P, obs_p)
                kps3d[k] = [round(float(v), 1) for v in X]
        if any(p is not None for p in kps3d):
            frames_out.append({"f": f, "kps": kps3d})

    out_path = OUT / "human_track.json"
    out_path.write_text(json.dumps({"frames": frames_out}))
    print(f"wrote {out_path}  ({len(frames_out)} frames)")


if __name__ == "__main__":
    main()
