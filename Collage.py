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
FRAMES_DIR = os.path.join(BASE, "frames")
os.makedirs(FRAMES_DIR, exist_ok=True)

# --------------------------------------------------------
# Detect videos
# --------------------------------------------------------
VIDEOS = sorted(glob(os.path.join(BASE, "*.mp4")))
if len(VIDEOS) != 4:
    raise RuntimeError(f"Expected 4 videos, found {len(VIDEOS)}")

print("Using videos:")
for v in VIDEOS:
    print(" -", v)

# --------------------------------------------------------
# Probe video size
# --------------------------------------------------------
def probe(path):
    cmd = [
        FFPROBE,
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        path
    ]
    info = json.loads(subprocess.check_output(cmd))
    stream = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    return stream["width"], stream["height"]

sizes = [probe(v) for v in VIDEOS]
widths = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

W = min(widths)
H = min(heights)

print(f"\nChosen working resolution (min across videos): {W}x{H}\n")

# --------------------------------------------------------
# STEP 1 — EXTRACT PNG FRAMES PER CAMERA (FAST, USING FFMPEG)
# --------------------------------------------------------
print("Extracting individual PNG frames from each video...")

for video in VIDEOS:
    name = os.path.splitext(os.path.basename(video))[0]
    outdir = os.path.join(FRAMES_DIR, name)
    os.makedirs(outdir, exist_ok=True)

    cmd = [
        FFMPEG,
        "-y",
        "-i", video,
        "-vf", f"fps={FPS},scale={W}:{H}",
        os.path.join(outdir, "%06d.png")
    ]

    print(f" → Extracting {name} ...")
    subprocess.run(cmd)

print("PNG extraction completed.\n")

# --------------------------------------------------------
# STEP 2 — RAM-ONLY MULTIVIEW COLLAGE
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

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
thickness = 3
padding = 10

print("Building multiview video...")

frame_index = 0

while True:
    # Read next frame from all 4 videos
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

    # Add border around each frame
    bordered = [
        cv2.copyMakeBorder(f, border, border, border, border,
                           cv2.BORDER_CONSTANT, value=(0,0,0))
        for f in frames
    ]

    # Build 2×2 collage
    top = np.hstack(bordered[:2])
    bottom = np.hstack(bordered[2:])
    collage = np.vstack([top, bottom])

    # --------------------------------------------------------
    # Draw dark rectangle behind frame number
    # --------------------------------------------------------
    label = f"Frame {frame_index}"
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

    x = collage.shape[1] - tw - padding*2
    y = th + padding

    # dark background rectangle
    cv2.rectangle(
        collage,
        (x - padding, y - th - padding),
        (x + tw + padding, y + padding),
        (0,0,0),
        -1
    )

    # white text
    cv2.putText(collage, label, (x, y), font, font_scale, (255,255,255), thickness)

    writer.stdin.write(collage.tobytes())
    frame_index += 1

# Cleanup
writer.stdin.close()
writer.wait()

for r in readers:
    r.stdout.close()

print("\nDONE — Created multiview video:")
print(OUT_VIDEO)
print("Extracted frames in:")
print(FRAMES_DIR)