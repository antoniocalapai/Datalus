import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

BASE = "/Users/acalapai/Desktop/test"
OUT_FRAMES = os.path.join(BASE, "frames")
OUT_MULTI = os.path.join(BASE, "multiview")
os.makedirs(OUT_FRAMES, exist_ok=True)
os.makedirs(OUT_MULTI, exist_ok=True)

# ---------------------------------------------------------
# Detect whether step 1 has already been completed
# ---------------------------------------------------------
videos = sorted(glob(os.path.join(BASE, "*.mp4")))
print("Found videos:", videos)

frame_dirs = []
step1_needed = False

for vid in videos:
    name = os.path.splitext(os.path.basename(vid))[0]
    outdir = os.path.join(OUT_FRAMES, name)
    frame_dirs.append(outdir)

    pngs = glob(os.path.join(outdir, "*.png"))
    if len(pngs) == 0:   # no extracted frames found
        step1_needed = True

# ---------------------------------------------------------
# 1. Extract frames (only if needed)
# ---------------------------------------------------------
if step1_needed:
    print("Step 1: Extracting frames...")
    for vid, outdir in zip(videos, frame_dirs):
        name = os.path.splitext(os.path.basename(vid))[0]
        os.makedirs(outdir, exist_ok=True)

        cap = cv2.VideoCapture(vid)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        pbar = tqdm(total=total_frames, desc=f"Extracting {name}", unit="frame")

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(os.path.join(outdir, f"{idx:06d}.png"), frame)
            idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        print(f"Extracted {idx} frames from {vid}")
else:
    print("Step 1 skipped — existing PNG frames detected.")

# ---------------------------------------------------------
# 2. Create multiview 2x2 collage frames (with black borders)
# ---------------------------------------------------------
print("Loading extracted frames...")
frame_lists = [sorted(glob(os.path.join(d, "*.png"))) for d in frame_dirs]
num_frames = min(len(fl) for fl in frame_lists)

print("Creating multiview frames...")

border = 10  # border thickness in pixels

for i in tqdm(range(num_frames), desc="Building multiview", unit="frame"):
    imgs = [cv2.imread(frame_lists[k][i]) for k in range(4)]

    # Resize to match the first image
    h, w = imgs[0].shape[:2]
    imgs_resized = [cv2.resize(im, (w, h)) for im in imgs]

    # Add black border around each image
    imgs_bordered = [
        cv2.copyMakeBorder(
            im, border, border, border, border,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0)  # black color
        )
        for im in imgs_resized
    ]

    # Update sizes after border
    h2, w2 = imgs_bordered[0].shape[:2]

    # Build collage
    top = cv2.hconcat(imgs_bordered[:2])
    bottom = cv2.hconcat(imgs_bordered[2:])
    collage = cv2.vconcat([top, bottom])

    cv2.imwrite(os.path.join(OUT_MULTI, f"{i:06d}.png"), collage)

# ---------------------------------------------------------
# 3. Convert multiview frames into a 5 FPS video
# ---------------------------------------------------------
multiview_frames = sorted(glob(os.path.join(OUT_MULTI, "*.png")))
example = cv2.imread(multiview_frames[0])
h, w = example.shape[:2]

fps = 5
out_video = os.path.join(BASE, "multiview.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

print("Writing multiview video...")

for f in tqdm(multiview_frames, desc="Saving video", unit="frame"):
    img = cv2.imread(f)
    writer.write(img)

writer.release()
print("Multiview video saved:", out_video)