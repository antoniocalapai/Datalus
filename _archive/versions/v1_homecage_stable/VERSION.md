# HomeCage v1.0 — Stable

**Date:** 2026-03-30

## What this is
Full end-to-end HomeCage pipeline: 4-camera 3D pose tracking of two rhesus macaques (Elm, Jok).

## Pipeline
- Script: `HomeCage_Calibration.py`
- Cameras: 102, 108, 113, 117
- Model: yolo26m-pose-ElmJok.pt (streaming inference, no frame extraction to disk)
- Detections: `pose_{cam_id}.txt` (one detection per animal per frame)
- Calibration: recoverPose + 2-camera Procrustes (gravity constraint) + look-at for 113/117
- 3D tracking: pairwise DLT, hip midpoint (kps 11+12), RANSAC 300mm, 5-frame rolling median
- Viewer: Three.js, PriCaB-identical styling, 2×2 camera strip, CoM spheres + skeleton

## Key results
- Reprojection error: 14.5px (anchor pair 102+108)
- Camera positions: exact known positions (Procrustes alignment)
- monkey (Elm): 3140/10839 frames triangulated (29.0%)
- monkey_0 (Jok): 27 frames
- Viewer: `homecage_viewer.html` (19 MB, proxy videos at HomeCage_output/proxy/)

## Files
- `HomeCage_Calibration.py` — full pipeline script
- `homecage_viewer.html` — interactive 3D viewer (requires HTTP server + proxy videos)
- `pose_{cam_id}.txt` — 2D detections per camera
- `yamls/` — per-camera calibration (K, R, T) in ABT format
