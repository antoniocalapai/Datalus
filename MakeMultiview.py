import subprocess
import json
import numpy as np
import cv2
import os
import re
from glob import glob

BASE = "/Users/acalapai/Desktop/Collage"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
FPS = 5

# Output folder for multiview
MULTIVIEW_DIR = os.path.join(BASE, "multiview")
os.makedirs(MULTIVIEW_DIR, exist_ok=True)

# --------------------------------------------------------
# Detect videos
# --------------------------------------------------------
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if len(VIDEOS) != 4:
    raise RuntimeError(f"Expected 4 videos, found {len(VIDEOS)}")

print("\nUsing videos:")
for v in VIDEOS:
    print(" -", v)

# --------------------------------------------------------
# Extract camera IDs + date
# --------------------------------------------------------
def extract_cam_id(path):
    name = os.path.basename(path)
    m = re.search(r"__([0-9]{3})_", name)
    return m.group(1) if m else None

def extract_date(path):
    name = os.path.basename(path)
    m = re.search(r"(\d{8})\d{6}", name)
    return m.group(1) if m else "00000000"

CAMERA_IDS   = [extract_cam_id(v) for v in VIDEOS]
CAMERA_DATES = [extract_date(v) for v in VIDEOS]
SESSION_DATE = CAMERA_DATES[0]

print("\nDetected camera IDs:", CAMERA_IDS)
print("Detected session date:", SESSION_DATE)

# --------------------------------------------------------
# Probe resolution
# --------------------------------------------------------
def probe(path):
    cmd = [
        FFPROBE, "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]
    info = json.loads(subprocess.check_output(cmd))
    stream = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    return stream["width"], stream["height"]

sizes = [probe(v) for v in VIDEOS]
W = min(s[0] for s in sizes)
H = min(s[1] for s in sizes)

print(f"\nChosen working resolution: {W}x{H}\n")

# --------------------------------------------------------
# Build multiview video
# --------------------------------------------------------
OUT_VIDEO = os.path.join(MULTIVIEW_DIR, f"{SESSION_DATE}_multiview.mp4")

def start_reader(video):
    cmd = [
        FFMPEG,
        "-i", video,
        "-vf", f"fps={FPS},scale={W}:{H}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE)

readers = [start_reader(v) for v in VIDEOS]

border = 10
collage_W = W * 2 + 40
collage_H = H * 2 + 40

writer = subprocess.Popen(
    [
        FFMPEG, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{collage_W}x{collage_H}",
        "-r", str(FPS),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        OUT_VIDEO,
    ],
    stdin=subprocess.PIPE
)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
padding = 10

print("Building multiview video...\n")

frame_index = 0

while True:

    frames = []
    for r in readers:
        raw = r.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            frames = []
            break
        frames.append(np.frombuffer(raw, np.uint8).reshape((H, W, 3)))

    if len(frames) < 4:
        break

    bordered = [
        cv2.copyMakeBorder(f, border, border, border, border,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
        for f in frames
    ]

    top = np.hstack(bordered[:2])
    bottom = np.hstack(bordered[2:])
    collage = np.vstack([top, bottom])

    label = f"Frame {frame_index}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = collage.shape[1] - tw - padding*2
    y = th + padding

    cv2.rectangle(collage, (x-padding, y-th-padding),
                  (x+tw+padding, y+padding), (0,0,0), -1)
    cv2.putText(collage, label, (x, y), font,
                font_scale, (255,255,255), thickness)

    writer.stdin.write(collage.tobytes())
    frame_index += 1

writer.stdin.close()
writer.wait()

for r in readers:
    r.stdout.close()

print("\nDONE — Created multiview video:")
print(OUT_VIDEO)