#!/usr/bin/env python3
"""
HomeCagePaper · make_example_video.py

Build a 4-camera example video for the presentation.
For a chosen session and frame range, draws the RT bounding boxes and 17
keypoints on every frame, composites the four cameras into a 2×2 grid, and
saves an mp4 to figures/example_video.mp4.

Run:
    python3 HomeCagePaper/make_example_video.py
"""
import cv2
import csv
import sys
import numpy as np
from pathlib import Path

HERE   = Path(__file__).parent
sys.path.insert(0, str(HERE))
from _style import PAL_WONG                              # noqa: E402

# ── Pick what to render ─────────────────────────────────────────────────────
SESSION    = "250711"
# Best 60-s "moving + clean" window with locally-available video:
#   28 inference issues in 60 s, median speed ~20 cm/s,
#   Jok tracked by 3.6 cams on average (best Jok coverage in moving windows).
CLIPS = [
    (548, 848),
]
SEPARATOR_FRAMES = 5            # 1 s blank separator between clips
RECORDING_FPS = 5.0             # the cameras record at 5 fps
OUT_FPS       = RECORDING_FPS
OUT_FILE   = HERE / "figures" / "example_video.mp4"
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CAMS = [102, 108, 113, 117]
KP_NAMES = [
    "nose","left_eye","right_eye","left_ear","right_ear",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle",
]
COCO_PAIRS = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]

# Per-monkey BGR colours (from the Wong palette, RGB → BGR)
def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

ANIMAL_BGR = {
    "Elm": hex_to_bgr(PAL_WONG["Elm"]),   # blue
    "Jok": hex_to_bgr(PAL_WONG["Jok"]),   # orange
}

_RAW_DIR = HERE / "data" / "sessions" / SESSION / "RAW"
VIDEOS = {
    102: _RAW_DIR / "July_2025__102_20250711151000.mp4",
    108: _RAW_DIR / "July_2025__108_20250711151000.mp4",
    113: _RAW_DIR / "July_2025__113_20250711151000.mp4",
    117: _RAW_DIR / "July_2025__117_20250711151000.mp4",
}
DETECTIONS_DIR = HERE / "data" / "sessions" / SESSION              # RT model
ABT_DIR        = None                                              # no ABT for 250713


def in_any_clip(f_idx):
    return any(s <= f_idx <= e for s, e in CLIPS)


def load_abt_detections(cam):
    """Parse the space-separated ABT _2D_result.txt for one camera.
    Returns {frame: [(animal, bbox, kps), ...]} restricted to clip windows."""
    if ABT_DIR is None or not ABT_DIR.exists():
        return {}
    matches = list(ABT_DIR.glob(f"*_{cam}_*_2D_result.txt"))
    out = {}
    if not matches:
        return out
    path = matches[0]
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) < 58:
                continue
            try:
                f_idx = int(parts[0])
            except ValueError:
                continue
            if not in_any_clip(f_idx):
                continue
            animal = parts[1]
            if animal not in ANIMAL_BGR:
                continue
            try:
                bbox = (float(parts[2]), float(parts[3]),
                        float(parts[4]), float(parts[5]))
            except ValueError:
                continue
            kps = []
            for k in range(17):
                base = 7 + k * 3
                try:
                    x, y, c = (float(parts[base]), float(parts[base+1]),
                               float(parts[base+2]))
                    kps.append((x, y, c))
                except (ValueError, IndexError):
                    kps.append((0.0, 0.0, 0.0))
            out.setdefault(f_idx, []).append((animal, bbox, kps))
    return out


# ── Load detections per camera into {frame: list of (animal, bbox, kps)} ────
def load_detections(cam):
    path = DETECTIONS_DIR / f"detections_cam{cam}.txt"
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            f_idx = int(row["frame_number"])
            if not in_any_clip(f_idx):
                continue
            animal = row.get("monkey_name", "")
            if animal not in ANIMAL_BGR:
                continue
            try:
                bbox = (
                    float(row["bbox_x1"]), float(row["bbox_y1"]),
                    float(row["bbox_x2"]), float(row["bbox_y2"]),
                )
            except (KeyError, ValueError):
                continue
            kps = []
            for name in KP_NAMES:
                sx = row.get(f"{name}_x", "")
                sy = row.get(f"{name}_y", "")
                sc = row.get(f"{name}_confidence", "")
                if sx.strip() and sy.strip():
                    c = float(sc) if sc.strip() else 1.0
                    kps.append((float(sx), float(sy), c))
                else:
                    kps.append((0.0, 0.0, 0.0))
            out.setdefault(f_idx, []).append((animal, bbox, kps))
    return out


# ── Drawing ─────────────────────────────────────────────────────────────────
def draw_overlay(frame, dets):
    """In-place drawing of bboxes + keypoints + skeleton."""
    for animal, bbox, kps in dets:
        color = ANIMAL_BGR[animal]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        # Animal name above bbox
        label = animal
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)[0]
        cv2.rectangle(frame, (x1, y1 - ts[1] - 16),
                      (x1 + ts[0] + 12, y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        # Skeleton bones
        for a, b in COCO_PAIRS:
            ka, kb = kps[a], kps[b]
            if ka[2] > 0.3 and kb[2] > 0.3:
                cv2.line(frame, (int(ka[0]), int(ka[1])),
                                (int(kb[0]), int(kb[1])), color, 3)
        # Keypoint dots
        for x, y, c in kps:
            if c > 0.3:
                cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), -1)
                cv2.circle(frame, (int(x), int(y)), 6, color, 2)


def add_cam_label(frame, cam):
    txt = f"cam {cam}"
    cv2.rectangle(frame, (20, 20), (260, 90), (0, 0, 0), -1)
    cv2.putText(frame, txt, (38, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)


def add_frame_label(frame, frame_idx, clip_idx, n_clips):
    txt = f"clip {clip_idx}/{n_clips}  ·  frame {frame_idx}"
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (w - 720, 20), (w - 20, 90), (0, 0, 0), -1)
    cv2.putText(frame, txt, (w - 700, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)


def make_separator(canvas_w, canvas_h, clip_idx, n_clips, start, end):
    """Solid black frame announcing the next clip."""
    img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    line1 = f"clip {clip_idx} / {n_clips}"
    line2 = f"session {SESSION}  ·  frames {start}-{end}"
    line3 = f"{(end-start)/RECORDING_FPS:.0f} s"
    for txt, y, scale in ((line1, canvas_h//2 - 80, 2.4),
                          (line2, canvas_h//2,       1.4),
                          (line3, canvas_h//2 + 80,  1.6)):
        ts = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)[0]
        x = (canvas_w - ts[0]) // 2
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 3)
    return img


# ── Main pipeline ───────────────────────────────────────────────────────────
def main():
    print(f"Session {SESSION}, {len(CLIPS)} clips:")
    for i, (s, e) in enumerate(CLIPS, 1):
        print(f"  clip {i}: frames {s}–{e}  ({(e-s)/RECORDING_FPS:.1f} s)")

    # Open video captures
    caps = {}
    for cam in CAMS:
        path = VIDEOS[cam]
        if not path.exists():
            print(f"  cam{cam}: missing video {path}"); continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"  cam{cam}: failed to open"); continue
        caps[cam] = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"  cam{cam}: {w}×{h}  {n} frames")

    if len(caps) != 4:
        print("Need all 4 cameras"); sys.exit(1)

    # Load detections — RT (live model) and ABT (yolo) for the same windows
    print("\nLoading RT detections…")
    rt_by_cam = {}
    for cam in CAMS:
        d = load_detections(cam)
        rt_by_cam[cam] = d
        print(f"  cam{cam}: {sum(len(v) for v in d.values())} dets")

    print("\nLoading ABT detections…")
    abt_by_cam = {}
    for cam in CAMS:
        d = load_abt_detections(cam)
        abt_by_cam[cam] = d
        print(f"  cam{cam}: {sum(len(v) for v in d.values())} dets")

    # Determine output canvas. If we have ABT detections, render TWO 2×2 grids
    # side by side; otherwise just one 2×2 grid (RT only).
    have_abt = any(abt_by_cam[c] for c in CAMS)
    sample = caps[CAMS[0]]
    cell_w = int(sample.get(cv2.CAP_PROP_FRAME_WIDTH))
    cell_h = int(sample.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = 0.30 if have_abt else 0.50
    cell_w_s = int(cell_w * scale)
    cell_h_s = int(cell_h * scale)
    grid_w = cell_w_s * 2
    grid_h = cell_h_s * 2
    title_h = 60 if have_abt else 0
    canvas_w = grid_w * (2 if have_abt else 1)
    canvas_h = grid_h + title_h
    print(f"\nOutput canvas: {canvas_w}×{canvas_h}  (have_abt={have_abt})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_FILE), fourcc, OUT_FPS,
                             (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"Failed to open writer at {OUT_FILE}"); sys.exit(1)

    def title_strip():
        strip = np.zeros((title_h, canvas_w, 3), dtype=np.uint8)
        for txt, x in (("ABT (yolo, jittery / stable ID)", 30),
                       ("RT (live model, smooth / ID jumps)", grid_w + 30)):
            cv2.putText(strip, txt, (x, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.line(strip, (grid_w, 0), (grid_w, title_h), (255, 255, 255), 2)
        return strip

    def render_grid(f_idx, dets_by_cam, clip_i, draw_frame_label):
        cells = {}
        for ci, cam in enumerate(CAMS):
            cap = caps[cam]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            dets = dets_by_cam[cam].get(f_idx, [])
            draw_overlay(frame, dets)
            add_cam_label(frame, cam)
            if ci == 0 and draw_frame_label:
                add_frame_label(frame, f_idx, clip_i, len(CLIPS))
            cells[cam] = cv2.resize(frame, (cell_w_s, cell_h_s))
        top    = np.hstack([cells[102], cells[108]])
        bottom = np.hstack([cells[113], cells[117]])
        return np.vstack([top, bottom])

    n_frames_total = sum(e - s + 1 for s, e in CLIPS)
    print(f"\nWriting {n_frames_total} clip frames + {len(CLIPS) * SEPARATOR_FRAMES} separator frames…")

    title = title_strip() if have_abt else None
    written = 0
    for clip_i, (start, end) in enumerate(CLIPS, 1):
        sep_inner = make_separator(canvas_w, grid_h, clip_i, len(CLIPS), start, end)
        sep = np.vstack([title, sep_inner]) if have_abt else sep_inner
        for _ in range(SEPARATOR_FRAMES):
            writer.write(sep)

        for f_idx in range(start, end + 1):
            grid_rt = render_grid(f_idx, rt_by_cam, clip_i, draw_frame_label=True)
            if have_abt:
                grid_abt = render_grid(f_idx, abt_by_cam, clip_i,
                                       draw_frame_label=False)
                body = np.hstack([grid_abt, grid_rt])
                canvas = np.vstack([title, body])
            else:
                canvas = grid_rt
            writer.write(canvas)
            written += 1
            if written % 50 == 0:
                print(f"  rendered {written}/{n_frames_total} clip frames")

    writer.release()
    for cap in caps.values():
        cap.release()
    print(f"\nSaved → {OUT_FILE}  ({OUT_FILE.stat().st_size/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
