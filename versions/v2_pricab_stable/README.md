# PriCaB v2 — Stable Snapshot

**Date:** 2026-03-28
**Git tag:** `v2-pricab-stable`
**Pipeline script:** `PriCaB_HumanCalib.py` (frozen copy)

---

## Regeneration command

```bash
# From Datalus/ root — deletes pose cache to force full re-inference
rm -f PriCaB_output/pose_*.txt PriCaB_output/pricab_poses*.npz PriCaB_output/pricab_viewer.html
python3 PriCaB_HumanCalib.py Measurements/250404_HumanTest_2
```

---

## What works

| Fix | Status | Notes |
|-----|--------|-------|
| 1. YOLO11x-pose inference | ✓ | Parallel (8 workers). Confirmed same counts as pre-existing yolo11x run. |
| 2. Keypoint overlay alignment | ✓ | Per-axis scale (sx=cw/iw, sy=ch/ih). Image stretches to fill tile; keypoints track correctly. |
| 3. Wall-topology room alignment | ✓ | Multi-start Nelder-Mead (72 starts, global minimum). Validation table printed. |
| 4. 3D skeleton quality | ✓ | Best-pair only, reprojection < 50 px, limb < 800 mm, 5-frame median. |
| 5. 10 fps viewer frames | ✓ | 100 ms intervals, 410 frames/camera, pre-extracted in `frames_viewer/`. |
| 6. Frame sync | ✓ | `Math.floor` mapping; global offset slider −10…+10; per-camera offset inputs. |
| 7. CoM marker | ✓ | Plotly size=20, white outline width=2. |
| 8. UI controls | ✓ | Skeleton/keypoints/CoM toggles, speed 0.25×–2×, MediaRecorder record button. |
| 9. Layout | ✓ | 4×4 tile grid left, 3D plot right, fills browser, single slider. |

---

## Known limitations

- **cam113 not on Wall A**: multi-start optimizer cannot align cam113 (viewer_X=2177 mm) with
  Wall A cameras (106=88, 107=541, 105=407, 110=−88 mm). The reconstruction places cam113 ~2 m
  off the expected wall. Root cause: only 17 solvePnP inliers; position unreliable. Wall B, C, D
  and ground level align well (std 14–276 mm).
- **Cameras 101, 104, 115 not placed**: insufficient landmark overlap with anchor pair (111↔114).
  Identity pose used; excluded from 3D viewer.
- **Intrinsics**: only cam102 and cam113 have checkerboard intrinsics. Other 14 cameras use
  thin-lens estimate (fx=1463 px, principal point at centre).

---

## Results summary

| Metric | Value |
|--------|-------|
| Cameras placed | 13 / 16 |
| Reprojection error (post-BA) | 14.5 px |
| Ground cameras height variance | std=14 mm ← excellent |
| Wall B X variance | std=240 mm ← good |
| Wall A X variance | std=807 mm ← dominated by cam113 outlier |
| Viewer file size | 55.4 MB |

---

## Wall groups used in alignment

| Group | Camera IDs | Aligned axis |
|-------|-----------|-------------|
| Wall A (origin wall, X≈0) | 113, 106, 107, 105, 110 | viewer_X |
| Wall B (far wall, X≈max) | 118, 112, 109, 114, 102 | viewer_X |
| Wall C (Y≈0 side)         | 106, 110, 112, 118, 119 | viewer_Y |
| Wall D (Y≈max side)       | 102, 109, 105, 107, 111 | viewer_Y |
| Ground (Z≈floor)          | 106, 107, 112, 109      | viewer_Z |
