# PriCaB v2 — Stable Snapshot

**Date:** 2026-03-28
**Tag:** v2-pricab-stable
**Pipeline script:** `PriCaB_HumanCalib.py` (frozen copy)

---

## What changed from v1

| Fix | Description |
|-----|-------------|
| 1 | **Wall-topology room alignment** — replaced heuristic yaw + SVD tilt with `scipy.optimize.minimize` (Nelder-Mead). Step 1 minimises variance of viewer_X within Wall A/B and viewer_Y within Wall C/D (yaw). Step 2 minimises variance of raw_Y within ground cameras (tilt Rx+Rz). Step 3 translates origin to centroid(106,110). |
| 2 | **Skeleton quality filter** — `KP_CONF_DRAW` raised to 0.5 (matches 3-D threshold); connections skipped when either endpoint is below threshold. |
| 3 | **Triangulation reprojection filter** — reprojection error < 50 px required from both cameras; best-scoring pair selected per keypoint. |
| 4 | **CoM marker** — size=20, white outline width=2 (was size=10 / width=1). |
| 5 | **All UI controls** — skeleton toggle, keypoints toggle, CoM-only mode, Record (MediaRecorder webm), speed 0.25×/0.5×/1×/2×. |
| 6 | **4×4 tile layout** — canvas letterbox (object-fit: contain behaviour) fills each cell. |
| 7 | **10 fps viewer frames** — extracted at 100 ms intervals to `frames_viewer/`. |
| 8 | **Frame sync** — `Math.floor` instead of `Math.round`; global offset slider (−10…+10); per-camera offset input field next to each tile label. |
| 9 | **This freeze.** |

## Wall groups used in alignment

| Group | Camera IDs | Axis |
|-------|-----------|------|
| Wall A (origin wall) | 113, 106, 107, 105, 110 | viewer X ≈ 0 |
| Wall B (far wall)    | 118, 112, 109, 114, 102  | viewer X ≈ max |
| Wall C (side wall 0) | 106, 110, 112, 118, 119  | viewer Y ≈ 0 |
| Wall D (side wall max)| 102, 109, 105, 107, 111 | viewer Y ≈ max |
| Ground               | 106, 107, 112, 109       | viewer Z ≈ low |

## Results

- **Cameras placed:** 13 / 16 (101, 104, 115 failed — insufficient landmark overlap)
- **Median reprojection error (post-BA):** 14.5 px
- **Ground cameras raw-Y (should be equal):** 106=1485, 107=1458, 112=1458, 109=1486 mm
- **Viewer:** `pricab_viewer.html` (55.4 MB, self-contained)
- **YAMLs:** `yamls/<cam_id>.yaml` (ABT-compatible)
