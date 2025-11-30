#!/usr/bin/env python3
import subprocess
import json
import numpy as np
import cv2
import os
import re
from glob import glob
from datetime import datetime, timedelta

# ================================================================
# CONFIG
# ================================================================
BASE = "/Users/acalapai/Desktop/Collage/RAW/250711"   # ← RAW folder
OUT_BASE = "/Users/acalapai/Desktop/Collage/multiview"  # ← where collage goes

FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"

# Preview length in minutes (1 = 1 minute, 0 = full video)
PREVIEW_MINUTES = 1

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1

TEXT_COLOR = (255, 255, 255)  # white
TEXT_BG = (0, 0, 0)           # black

# How much to downscale from the smallest original resolution
SCALE_FACTOR = 0.5   # 0.5 = half-size

# ================================================================
# LOAD VIDEOS
# ================================================================
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))

if len(VIDEOS) != 4:
    raise RuntimeError(f"Expected 4 mp4 files inside RAW/250711, found {len(VIDEOS)}")

print("\nUsing videos:")
for v in VIDEOS:
    print(" -", v)

# ================================================================
# CAMERA IDs + SESSION TIMESTAMP
# ================================================================
def extract_cam_id(path):
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else "???"

def extract_start_timestamp(path):
    """
    Extract 14-digit timestamp (YYYYMMDDHHMMSS) from filename.
    """
    name = os.path.basename(path)
    m = re.findall(r"(\d{14})", name)
    return m[0] if m else None

CAMERA_IDS = [extract_cam_id(v) for v in VIDEOS]
RECORDING_TS = extract_start_timestamp(VIDEOS[0])
if RECORDING_TS is None:
    raise RuntimeError("Could not parse timestamp from RAW filenames.")

SESSION_START = datetime.strptime(RECORDING_TS, "%Y%m%d%H%M%S")
SESSION_DATE = RECORDING_TS[:8]

# ================================================================
# PROBE RESOLUTION & FPS
# ================================================================
def probe_stream_info(path):
    info = json.loads(subprocess.check_output([
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]))
    return [s for s in info["streams"] if s["codec_type"] == "video"][0]

def probe_resolution(path):
    s = probe_stream_info(path)
    return s["width"], s["height"]

def probe_fps(path):
    s = probe_stream_info(path)

    def _rate_to_float(rate_str):
        if not rate_str or rate_str == "0/0":
            return 0.0
        num, den = rate_str.split("/")
        num = float(num)
        den = float(den)
        return num / den if den > 0 else 0.0

    fps = _rate_to_float(s.get("r_frame_rate", "0/0"))
    if fps <= 0:
        fps = _rate_to_float(s.get("avg_frame_rate", "0/0"))
    if fps <= 0:
        raise RuntimeError("FPS missing in video metadata!")
    return fps

# Smallest original resolution (ensures matching scale)
sizes = [probe_resolution(v) for v in VIDEOS]
orig_W = min(s[0] for s in sizes)
orig_H = min(s[1] for s in sizes)
print(f"\nOriginal smallest resolution: {orig_W}x{orig_H}")

# ================================================================
# DOWNSCALING
# ================================================================
W = int(orig_W * SCALE_FACTOR)
H = int(orig_H * SCALE_FACTOR)
W = W // 2 * 2
H = H // 2 * 2
print(f"Downscaled resolution used for ALL cameras: {W}x{H}")

# FPS
FPS_FLOAT = probe_fps(VIDEOS[0])
FPS = int(round(FPS_FLOAT))
print(f"Detected FPS: {FPS_FLOAT:.3f} → using {FPS} fps\n")

# PREVIEW
if PREVIEW_MINUTES > 0:
    PREVIEW_FRAMES = int(PREVIEW_MINUTES * 60 * FPS)
else:
    PREVIEW_FRAMES = None

# ================================================================
# FFmpeg Readers
# ================================================================
def start_reader(video):
    cmd = [
        FFMPEG,
        "-i", video,
        "-vf", f"scale={W}:{H},fps={FPS}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)

readers = [start_reader(v) for v in VIDEOS]

# ================================================================
# Writer
# ================================================================
coll_W = W * 2 + 40
coll_H = H * 2 + 40

os.makedirs(OUT_BASE, exist_ok=True)

OUT_NAME = (
    f"{SESSION_DATE}_multiview_RAW_preview.mp4"
    if PREVIEW_MINUTES > 0 else
    f"{SESSION_DATE}_multiview_RAW.mp4"
)

OUT_VIDEO = os.path.join(OUT_BASE, OUT_NAME)

writer = subprocess.Popen(
    [
        FFMPEG, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{coll_W}x{coll_H}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        OUT_VIDEO
    ],
    stdin=subprocess.PIPE
)

# ================================================================
# HELPERS
# ================================================================
def draw_text_with_bg(img, text, pos):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    x, y = pos
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 8, y + 8), TEXT_BG, -1)
    cv2.putText(img, text, (x + 4, y), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS)

# ================================================================
# MAIN LOOP
# ================================================================
print("Building multiview video...\n")
frame_idx = 0

while True:
    if PREVIEW_FRAMES is not None and frame_idx >= PREVIEW_FRAMES:
        break

    frames = []
    for r in readers:
        raw = r.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            frames = []
            break
        frames.append(np.frombuffer(raw, np.uint8).reshape((H, W, 3)))

    if len(frames) < 4:
        break

    bordered = [cv2.copyMakeBorder(f, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0,0,0))
                for f in frames]

    top = np.hstack(bordered[:2])
    bottom = np.hstack(bordered[2:])
    collage = np.vstack([top, bottom])

    time_now = SESSION_START + timedelta(seconds=frame_idx / FPS)
    elapsed = time_now - SESSION_START

    time_of_day_str = time_now.strftime("%H:%M:%S")

    total_secs = int(elapsed.total_seconds())
    h_ = total_secs // 3600
    m_ = (total_secs % 3600) // 60
    s_ = total_secs % 60
    time_of_session_str = f"{h_:02d}:{m_:02d}:{s_:02d}"

    draw_text_with_bg(collage, f"Session {SESSION_DATE}   |   Frame {frame_idx}", (20, 50))
    draw_text_with_bg(collage, f"Session Time: {time_of_session_str}", (20, 80))
    draw_text_with_bg(collage, f"Clock Time:   {time_of_day_str}", (20, 110))

    cam_positions = [
        (20,       H - 20),
        (W + 60,   H - 20),
        (20,       H + H + 20),
        (W + 60,   H + H + 20)
    ]
    for ci, cid in enumerate(CAMERA_IDS):
        draw_text_with_bg(collage, f"Cam {cid}", cam_positions[ci])

    writer.stdin.write(collage.tobytes())
    frame_idx += 1

# ================================================================
# CLEANUP
# ================================================================
writer.stdin.close()
writer.wait()

for r in readers:
    try:
        r.stdout.close()
        r.terminate()
    except:
        pass

print("\nDONE — Created:")
print(OUT_VIDEO)