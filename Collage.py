import subprocess
import json
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

BASE = "/Users/acalapai/Desktop/test"
FFMPEG = "/opt/homebrew/bin/ffmpeg"
FFPROBE = "/opt/homebrew/bin/ffprobe"
FPS = 5
OUT_VIDEO = os.path.join(BASE, "multiview.mp4")

# --------------------------------------------------------
# Load 4 videos
# --------------------------------------------------------
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if len(VIDEOS) != 4:
    raise RuntimeError(f"Expected 4 videos, found {len(VIDEOS)}")

print("Using videos:")
for v in VIDEOS:
    print(" -", v)

# --------------------------------------------------------
# Get resolution using ffprobe JSON reliably
# --------------------------------------------------------
def probe(path):
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    info = json.loads(p.stdout.read())
    stream = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    return stream["width"], stream["height"]

sizes = [probe(v) for v in VIDEOS]
widths = [w for (w,h) in sizes]
heights = [h for (w,h) in sizes]

W = min(widths)   # force all 4 to same size
H = min(heights)

print(f"\nFinal chosen working resolution: {W}x{H}\n")

# --------------------------------------------------------
# Start readers (raw RGB24 frames, resized by ffmpeg)
# --------------------------------------------------------
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

# --------------------------------------------------------
# Setup writer
# --------------------------------------------------------
border = 10
collage_W = W*2 + 40
collage_H = H*2 + 40

writer = subprocess.Popen(
    [
        FFMPEG,
        "-y",
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

# --------------------------------------------------------
# Process frames
# --------------------------------------------------------
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 3

# We don't know exact frame count (ffprobe duration unreliable with MPEG4 SP)
# So we run until any stream ends
frame_index = 0

print("Processing frames...")

while True:
    frames = []
    for r in readers:
        raw = r.stdout.read(W * H * 3)
        if len(raw) < W * H * 3:
            frames = []
            break
        img = np.frombuffer(raw, dtype=np.uint8).reshape((H, W, 3))
        frames.append(img)

    if len(frames) < 4:
        break

    # Add borders
    bordered = [
        cv2.copyMakeBorder(f, border, border, border, border,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
        for f in frames
    ]

    top = np.hstack(bordered[:2])
    bottom = np.hstack(bordered[2:])
    collage = np.vstack([top, bottom])

    # Frame number
    label = f"Frame {frame_index}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    x = collage.shape[1] - tw - 20
    y = th + 20

    cv2.putText(collage, label, (x+2, y+2), font, font_scale, (0,0,0), thickness+2)
    cv2.putText(collage, label, (x, y),     font, font_scale, (255,255,255), thickness)

    writer.stdin.write(collage.tobytes())

    frame_index += 1

# --------------------------------------------------------
# Cleanup
# --------------------------------------------------------
writer.stdin.close()
writer.wait()

for r in readers:
    r.stdout.close()

print("\nDONE — created:")
print(OUT_VIDEO)