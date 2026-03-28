# PriCaB — Primate Calibration via Human Body  (v1.0)

Marker-free, checkerboard-free multi-camera calibration using a human
body as the calibration object.  A person walks through the scene once;
the script produces metric camera poses in millimetres ready for ABT.

---

## What it does

| Stage | Name | Output |
|-------|------|--------|
| 1 | Inspect | Camera IDs, resolution, fps |
| 2 | Extract frames | 1 PNG per 500 ms per camera |
| 3 | YOLOv8-pose | 17-keypoint detections per frame |
| 4 | Intrinsics | K, dist from checkerboard NPZ or lens estimate |
| 5 | Extrinsics | Metric poses via anchor pair + solvePnP + bundle adjustment |
| 6 | YAMLs | ABT-compatible per-camera YAML |
| 7 | Viewer | Self-contained interactive HTML |
| 8 | Config | datalus_config.json updated |

---

## Quick start

```bash
python3 PriCaB_HumanCalib.py  Measurements/250404_HumanTest_2
```

- Point it at any folder containing one synchronised video per camera.
- Camera IDs are detected automatically from file names.
- The viewer opens in your browser when done.
- Re-running is safe: stages 2 and 3 are skipped if their output already
  exists.  Delete `PriCaB_output/` to start from scratch.

---

## Prerequisites

```bash
pip install opencv-python numpy scipy ultralytics
```

| Library | Minimum version | Notes |
|---------|----------------|-------|
| opencv-python | 4.8 | cv2.solvePnPRansac, triangulatePoints |
| numpy | 1.24 | |
| scipy | 1.11 | least_squares (bundle adjustment) |
| ultralytics | 8.0 | YOLOv8-pose; only needed for stage 3 |

YOLOv8 model weights (`yolov8n-pose.pt`) are downloaded automatically on
first run if not present.

---

## Inputs

### Video files
- One video per camera, all recorded simultaneously
- Any common format: `.mp4`, `.avi`, `.mov`, `.mkv`
- All videos must be in the same folder

### Intrinsics (optional but recommended)
Place `DatalusCalibration/intrinsics.npz` next to the script.
It must contain `<cam_id>_K` (3×3) and `<cam_id>_dist` (1×5) arrays,
produced by the DatulusCalib checkerboard pipeline (Steps 1–2).

If a camera is absent from the NPZ, a thin-lens estimate is used:
```
fx = (LENS_FOCAL_MM / SENSOR_WIDTH_MM) × image_width_px
```

---

## Outputs

```
PriCaB_output/
├── frames/
│   └── <cam_id>/
│       └── <cam_id>_frame_NNNNNN.png    # extracted frames
├── pose_<cam_id>.txt                     # YOLO detections
├── pricab_poses.npz                      # metric R, T per camera
├── pricab_poses_scaled.npz              # identical (backward compat)
├── yamls/
│   └── <cam_id>.yaml                    # ABT-compatible YAML
└── pricab_viewer.html                   # interactive 3-D viewer
datalus_config.json                      # updated camera positions
```

### YAML format (ABT-compatible)

```yaml
camera_matrix:
  rows: 3  cols: 3  data: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
dist_coeffs:
  rows: 1  cols: 5  data: [k1, k2, p1, p2, k3]
rotation_matrix:
  rows: 3  cols: 3  data: [...]      # R (world → camera)
translation_vector:
  rows: 3  cols: 1  data: [tx, ty, tz]   # in mm
image_size: [width, height]
```

---

## How the extrinsics work

### Why not recoverPose for all cameras?

`cv2.recoverPose` always returns `|T| = 1` (unit-scale baseline).
Chaining pairwise results in a "star topology" — every camera pair has
its own independent unit baseline — which gives wildly inconsistent
camera positions (reprojection errors > 5000 px).

### The anchor + solvePnP approach

1. **Anchor pair** — the camera pair with the most co-detected
   full-body frames.  `recoverPose` gives a unit-scale relative pose.

2. **Metric scale** — the known person height (PERSON_HEIGHT_MM = 1700 mm)
   is compared to the median head-to-ankle distance triangulated in raw
   units.  The scale factor is applied directly to the anchor translation
   so the entire reconstruction is in mm from this point on.

3. **Landmark triangulation** — SCALE_KPS (nose, shoulders, hips, ankles)
   are triangulated from the anchor pair into metric 3-D space.

4. **solvePnPRansac** — each remaining camera is placed by matching its
   2-D keypoint observations to the metric 3-D landmarks.  Cameras are
   sorted by landmark overlap so each solved camera can add new landmarks
   for the next (iterative expansion outward from the anchor).

5. **Bundle adjustment** — `scipy.optimize.least_squares` with TRF solver
   and soft-L1 loss jointly refines all non-anchor camera poses.  The 3-D
   landmarks are fixed (the person height constraint was already applied).

Typical results: 19–25 px median reprojection error on 16-camera rigs
with a single person walk.

### Room alignment

After reconstruction the coordinate frame is arbitrary (defined by the
anchor camera pair).  A rigid transform anchors it to the physical room:

1. **Translate** — the centroid of cameras 106 and 110 (origin corner)
   is moved to viewer (X=0, Y=0).  Height (Z) is unchanged.

2. **Rotate (yaw)** — the reconstruction is rotated around the vertical
   axis so that the line from cams 106/110 toward cams 105/107 points
   along positive viewer X (along the wall).

This does not change reprojection error — it is a pure rigid transform
applied equally to all camera positions and rotation matrices.

---

## Coordinate system

```
       Z (up)
       │
       │
       └──────── X (along wall: cams 106/110 → 105/107)
      /
     /
    Y (into room)
```

All distances in millimetres.  Camera origin corner (106+110) at (0, 0).

---

## Viewer

Open `PriCaB_output/pricab_viewer.html` in any modern browser — no
server required.

```bash
open PriCaB_output/pricab_viewer.html          # macOS
xdg-open PriCaB_output/pricab_viewer.html      # Linux
```

- Use the **Play / frame slider** to animate the hip trajectory.
- Rotate the 3-D scene by dragging; zoom with scroll wheel.
- Camera feeds on the right sync with the 3-D frame.
- Cameras that could not be placed are excluded automatically.

---

## Tunable constants

Edit these at the top of `PriCaB_HumanCalib.py`:

| Constant | Default | Meaning |
|----------|---------|---------|
| `PERSON_HEIGHT_MM` | 1700 | Known subject height (mm) |
| `SCALE_KPS` | [0,5,6,11,12,15,16] | Keypoints used for scale (COCO indices) |
| `SCALE_BBOX_CONF` | 0.7 | Min bbox confidence for full-body frames |
| `SCALE_KP_CONF` | 0.5 | Min keypoint confidence for scale frames |
| `SCALE_VERT_FRAC` | 0.30 | Min body vertical span / image height |
| `FRAME_INTERVAL_MS` | 500 | Frame sampling interval (ms) |
| `KP_CONF_THRESH` | 0.5 | Min keypoint confidence for correspondences |
| `BBOX_CONF_THRESH` | 0.3 | Min bbox confidence to keep a detection |
| `LENS_FOCAL_MM` | 8.0 | Fallback focal length (mm) |
| `SENSOR_WIDTH_MM` | 11.2 | Fallback sensor width (mm) |
| `IMG_SCALE` | 0.25 | Viewer image resize factor |
| `JPEG_QUALITY` | 40 | Viewer JPEG quality |

---

## Troubleshooting

**"No valid full-body pairs"**
The person was never fully visible in two cameras simultaneously.
Lower `SCALE_BBOX_CONF` to 0.5 or `SCALE_VERT_FRAC` to 0.15.

**Many cameras failed placement (0 solvePnP inliers)**
These cameras never co-appear with the anchor pair.  Check that all
cameras overlap temporally with the walk.  Failed cameras are excluded
from the viewer but their YAMLs are still written (identity pose).

**High reprojection error (> 40 px)**
- Check that `PERSON_HEIGHT_MM` matches the actual subject.
- Provide proper checkerboard intrinsics — the thin-lens fallback may
  be inaccurate for wide-angle or fisheye lenses.
- Increase `SCALE_BBOX_CONF` to use only clean full-body detections.

**Viewer shows wrong room orientation**
The room alignment uses cameras 106/110 as origin and 105/107 as X axis.
If your rig uses different camera IDs, update the `origin_ids` and
`xaxis_ids` lists in `stage5_extrinsics` (around line 900).

---

## Version history

| Version | Date | Notes |
|---------|------|-------|
| v1.0 | 2026-03-28 | First stable release.  Anchor+solvePnP+BA extrinsics, metric scale from person height, explicit room alignment to cams 106/110 origin and 105/107 X axis. |
