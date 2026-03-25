import os
import cv2
import csv
from glob import glob
from ultralytics import YOLO
import re

# ============================================================
# CONFIG
# ============================================================
RAW_BASE = "/Users/acalapai/Desktop/Collage/RAW/250711/"        # unprocessed videos
PROC_BASE = "/Users/acalapai/Desktop/Collage"                   # ABT processed videos + txt files
OUT_DIR  = "/Users/acalapai/Desktop/Collage/YOLOv9_Test"

MODEL_PATH  = "yolov9c.pt"
PREVIEW_MIN = 5
DOWNSAMPLE  = 0.5

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD YOLOv9
# ============================================================
print("\nLoading YOLOv9 model…")
model = YOLO(MODEL_PATH)

# ============================================================
# HELPERS
# ============================================================
def extract_cam_id(path):
    name = os.path.basename(path)

    m = re.match(r"(\d{3})_", name)    # RAW
    if m:
        return m.group(1)

    m = re.search(r"__([0-9]{3})_", name)  # PROCESSED
    if m:
        return m.group(1)

    return None


def find_abt_txt(base, cam):
    pattern = os.path.join(base, f"July_2025__{cam}_*_2D_result.txt")
    files = glob(pattern)
    if len(files) == 0:
        raise RuntimeError(f"ABT txt not found for camera {cam}")
    return files[0]


def load_abt_boxes(txt_path):
    """
    Load ABT bounding boxes as:
    abt_data[frame_idx] = (x1,y1,x2,y2)
    """
    abt_data = {}

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if parts[0] == "frame_number" or len(parts) < 7:
                continue

            try:
                frame_idx = int(parts[0])
                x1 = float(parts[3])
                y1 = float(parts[4])
                x2 = float(parts[5])
                y2 = float(parts[6])
            except ValueError:
                continue

            abt_data[frame_idx] = (x1, y1, x2, y2)

    return abt_data


def iou(boxA, boxB):
    """
    Compute IoU between two boxes in x1,y1,x2,y2 format.
    """
    if boxA is None or boxB is None:
        return 0.0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    if union <= 0:
        return 0.0

    return inter / union


def yolo_to_xyxy(det):
    """Convert YOLOv9 detection into x1,y1,x2,y2."""
    x1, y1, x2, y2 = det.boxes.xyxy[0].tolist()
    return (x1, y1, x2, y2)


def bar(prefix, i, total):
    width = 30
    r = min(1.0, i / float(total))
    fill = int(r * width)
    line = "#" * fill + "-" * (width - fill)
    print(f"\r{prefix} [{line}] {i}/{total}", end="", flush=True)


# ============================================================
# LOCATE RAW + PROCESSED
# ============================================================
raw_videos = sorted(glob(os.path.join(RAW_BASE, "*.mp4")))
proc_videos = sorted(glob(os.path.join(PROC_BASE, "*.mp4")))

raw_dict  = {extract_cam_id(v): v for v in raw_videos}
proc_dict = {extract_cam_id(v): v for v in proc_videos}

cams = sorted(set(raw_dict.keys()) & set(proc_dict.keys()))

print("\nMatched Cameras:")
for c in cams:
    print(f"  Cam {c}:")
    print(f"     RAW:       {raw_dict[c]}")
    print(f"     PROCESSED: {proc_dict[c]}")


# ============================================================
# MAIN LOOP — PER CAMERA
# ============================================================
for cam in cams:

    raw_path  = raw_dict[cam]
    proc_path = proc_dict[cam]
    txt_path  = find_abt_txt(PROC_BASE, cam)

    print("\n=====================================================")
    print(f"Processing camera {cam}")
    print(f"RAW:   {raw_path}")
    print(f"PROC:  {proc_path}")
    print(f"TXT:   {txt_path}")
    print("=====================================================\n")

    abt_boxes = load_abt_boxes(txt_path)

    cap_raw  = cv2.VideoCapture(raw_path)
    cap_proc = cv2.VideoCapture(proc_path)

    fps = cap_raw.get(cv2.CAP_PROP_FPS)
    total_frames_raw  = int(cap_raw.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_proc = int(cap_proc.get(cv2.CAP_PROP_FRAME_COUNT))

    limit_frames = int(PREVIEW_MIN * 60 * fps)
    limit_frames = min(limit_frames, total_frames_raw, total_frames_proc)

    out_video = os.path.join(OUT_DIR, f"cam_{cam}_YOLOv9_compare.mp4")
    out_csv   = os.path.join(OUT_DIR, f"cam_{cam}_IoU.csv")

    csv_rows = []
    out_writer = None
    frame_idx = 0

    while frame_idx < limit_frames:

        bar("Progress", frame_idx, limit_frames)

        ok1, raw_frame = cap_raw.read()
        ok2, proc_frame = cap_proc.read()
        if not (ok1 and ok2):
            break

        # Downsample
        if DOWNSAMPLE != 1.0:
            raw_small = cv2.resize(raw_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
            proc_small = cv2.resize(proc_frame, None, fx=DOWNSAMPLE, fy=DOWNSAMPLE)
        else:
            raw_small = raw_frame
            proc_small = proc_frame

        # ======================================================
        # YOLOv9 detection (IMPORTANT FIX: use proc_small)
        # ======================================================
        yres = model.predict(proc_small, device="mps", verbose=False)

        if len(yres[0].boxes) > 0:
            yolo_box = yolo_to_xyxy(yres[0])
        else:
            yolo_box = None

        # Resize ABT box if downsampled
        abt_box = abt_boxes.get(frame_idx, None)
        if abt_box and DOWNSAMPLE != 1.0:
            x1, y1, x2, y2 = abt_box
            abt_box = (
                x1 * DOWNSAMPLE,
                y1 * DOWNSAMPLE,
                x2 * DOWNSAMPLE,
                y2 * DOWNSAMPLE,
            )

        # Compute IoU
        iou_val = iou(abt_box, yolo_box)

        # Visualization
        yolo_vis = yres[0].plot()
        combined = cv2.hconcat([proc_small, yolo_vis])

        # Overlay IoU
        cv2.putText(
            combined,
            f"IoU: {iou_val:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # Init video writer
        if out_writer is None:
            H, W = combined.shape[:2]
            out_writer = cv2.VideoWriter(
                out_video,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (W, H)
            )

        out_writer.write(combined)

        # Save CSV row
        csv_rows.append([frame_idx, iou_val])

        frame_idx += 1

    # Close everything
    cap_raw.release()
    cap_proc.release()
    if out_writer:
        out_writer.release()

    # Write IoU CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "IoU"])
        writer.writerows(csv_rows)

    print(f"\n✔ Saved video: {out_video}")
    print(f"✔ Saved IoU CSV: {out_csv}\n")

print("\nDONE — YOLOv9 vs ABT comparisons generated.")