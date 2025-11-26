import os
import cv2
from glob import glob
from tqdm import tqdm

BASE = "/Users/acalapai/Desktop/test"
OUT_FRAMES = os.path.join(BASE, "frames")
OUT_MULTI = os.path.join(BASE, "multiview")
os.makedirs(OUT_FRAMES, exist_ok=True)
os.makedirs(OUT_MULTI, exist_ok=True)

# ---------------------------------------------------------
# 1. Extract frames from each video into separate directories
# ---------------------------------------------------------
videos = sorted(glob(os.path.join(BASE, "*.mp4")))
print("Found videos:", videos)

frame_dirs = []

for vid in videos:
    name = os.path.splitext(os.path.basename(vid))[0]
    outdir = os.path.join(OUT_FRAMES, name)
    os.makedirs(outdir, exist_ok=True)
    frame_dirs.append(outdir)

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

# ---------------------------------------------------------
# 2. Create multiview 2x2 collage frames
# ---------------------------------------------------------
frame_lists = [sorted(glob(os.path.join(d, "*.png"))) for d in frame_dirs]
num_frames = min(len(fl) for fl in frame_lists)

print("Creating multiview frames...")

for i in tqdm(range(num_frames), desc="Building multiview", unit="frame"):
    imgs = [cv2.imread(frame_lists[k][i]) for k in range(4)]

    # Resize to the first image size
    h, w = imgs[0].shape[:2]
    imgs_resized = [cv2.resize(im, (w, h)) for im in imgs]

    top = cv2.hconcat(imgs_resized[:2])
    bottom = cv2.hconcat(imgs_resized[2:])
    collage = cv2.vconcat([top, bottom])

    cv2.imwrite(os.path.join(OUT_MULTI, f"{i:06d}.png"), collage)

# ---------------------------------------------------------
# 3. Convert multiview frames into a 5 FPS video
# ---------------------------------------------------------
multiview_frames = sorted(glob(os.path.join(OUT_MULTI, "*.png")))
example = cv2.imread(multiview_frames[0])
h, w = example.shape[:2]

fps = 5  # same as acquisition
out_video = os.path.join(BASE, "multiview.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

print("Writing multiview video...")

for f in tqdm(multiview_frames, desc="Saving video", unit="frame"):
    img = cv2.imread(f)
    writer.write(img)

writer.release()

print("Multiview video saved:", out_video)