# Datalus — Claude Project Configuration

## BEHAVIOR RULES — READ FIRST
- Never ask for permission before editing, creating, or running files
- Never ask "shall I proceed?" or "do you want me to?" — just do it
- Never ask for confirmation before installing packages
- Never stop mid-task to report progress and wait — run all stages end to end
- Only stop and ask if a required input file is genuinely missing and cannot be inferred
- Update datalus_config.json at the end of every task automatically
- Never modify anything in DatalusCalibration/, Measurements/, ABT_Software-main/, or CalibrationVideos/ unless explicitly told to

---

## PROJECT OVERVIEW

This is the Datalus project — a multi-camera 3D behavioral tracking system for non-human primates (rhesus macaques) at the Cognitive Neuroscience Laboratory, Deutsches Primatenzentrum (German Primate Center), Göttingen.

The system uses synchronized Hikrobot cameras feeding into NVIDIA Jetson boards for 2D inference, with a central server performing 3D triangulation. The software pipeline is called ABT (Autonomous Behavioral Tracking).

There are two active experimental environments:

### HomeCage
- 4 cameras: 102, 108, 113, 117
- 2 identified animals: Elm and Jok
- Room dimensions: x=2240, y=3400, z=3260 mm
- Camera positions (mm):
  - Cam102: (300, 3260, 540) — world reference origin
  - Cam108: (1850, 3240, 2480)
  - Cam113: (50, 0, 550)
  - Cam117: (2080, 3070, 2550)
- Coordinate alignment: cam113 at origin, Y axis toward cam102, X axis toward cam117
- 40 hours of session data (20 sessions × 2h) already recorded
- Pre-computed 2D keypoint results in Measurements/250711/2D_results/
- Extracted frames in DatalusCalibration/frames/{cam_id}/
- Goal: produce 3D trajectories for Elm and Jok, publish as Nature Communications proof-of-concept paper

### PriCaB (Primate Cage Behavior)
- 16 cameras: 101–107, 109–115, 118, 119
- Large open room, person walking through for calibration
- No known camera positions or room dimensions — everything derived from data
- Calibration approach: human pose keypoints from walking videos, person height = 1700mm for scale
- Coordinate alignment: cameras 106 and 110 define origin (0,0), cameras 105 and 107 define positive X axis
- Human calibration test data in Measurements/250404_HumanTest_2/

---

## CALIBRATION APPROACH

The calibration pipeline (called Datalus) produces ABT-compatible YAML files (K, dist, R, T per camera). No JARVIS, no COLMAP, no checkerboard stereo required.

### What works (do not redo these)
- Intrinsics: OpenCV checkerboard calibration via cv2.calibrateCamera — DONE, stored in DatalusCalibration/intrinsics.npz
  - RMS: cam102=1.3px, cam108=1.0px, cam113=4.2px, cam117=3.3px
- Extrinsics approach that works: animal/human body keypoints as stereo correspondences
  - Find synchronized frames where same subject detected in 2+ cameras
  - Use cv2.findEssentialMat + cv2.recoverPose with all 17 COCO keypoints
  - Chain weak camera pairs through best-connected intermediate (102→113→108)
  - Apply scale from known person height (1700mm) or known camera positions
  - Refine with cv2.solvePnPRansac using triangulated 3D world points
  - Bundle adjustment with scipy.optimize.least_squares as final refinement

### What failed (do not retry these)
- COLMAP on checkerboard frames — SIFT cannot match repeating patterns
- COLMAP on room frames — registered 0 images
- Checkerboard stereo (cv2.stereoCalibrate) — 0 shared frames, board never in shared volume
- Torso-only keypoints (indices 5,6,11,12) — insufficient spatial spread, geometry collapses

### Current best configuration
- All 17 keypoints, RANSAC inlier filtering
- Chain: 102→113→108 for HomeCage
- Scale: from known camera positions in mm (HomeCage) or person height 1700mm (PriCaB)
- HomeCage reprojection error: 219px (essmat) → improving with solvePnP + bundle adjustment
- PriCaB reprojection error: 292px → improving with scale recovery + bundle adjustment

---

## FILE STRUCTURE
```
Datalus/
├── datalus_config.json              — ground truth config, always update after tasks
├── DatalusCalibration/
│   ├── intrinsics.npz               — per-camera K and dist, DO NOT overwrite
│   ├── frames/{cam_id}/             — extracted PNG frames, DO NOT modify
│   ├── colmap_poses.npz             — current best extrinsics (essmat approach)
│   ├── colmap_poses_essmat.npz      — backup of essmat poses
│   └── yamls/{cam_id}.yaml          — current best YAML files
├── Measurements/
│   ├── 250711/2D_results/           — HomeCage ABT keypoint output for Elm and Jok
│   └── 250404_HumanTest_2/          — PriCaB human walking videos
├── CalibrationVideos/
│   ├── 250707/                      — session 1 calibration videos
│   └── 250708/                      — session 2 calibration videos
├── HomeCage_output/                 — all HomeCage pipeline outputs go here
├── PriCaB_output/                   — all PriCaB pipeline outputs go here
├── PoseModelBenchmark/              — pose model comparison outputs
├── HomeCage_Calibration.py          — HomeCage end-to-end calibration script
├── PriCaB_HumanCalib.py             — PriCaB single-script calibration
├── ScaleRecovery.py                 — scale recovery from person height
├── PoseModelBenchmark.py            — multi-model pose estimation comparison
├── DatulusCalib_Step3_PoseCorrespondences.py
├── DatulusCalib_Step4_WriteYAMLs.py
├── Datalus3D_Visualize.py
├── Datalus3D_Trajectory.py
└── build_viewer.py
```

---

## CAMERA AND HARDWARE SPECS
```
Model:           Hikrobot MV-CH120-10GC
Sensor:          Sony IMX304, 1.1" format
Sensor width:    11.2mm
Focal length:    8.0mm
Resolution:      2048×1496px (binning 2 — this is the operating resolution)
Focal length px: 1463px (computed: 8.0/11.2 × 2048)
Principal point: (1024, 748)
Keypoint format: COCO 17-point
```

---

## CHECKERBOARD SPECS
```
Inner corners: 13 wide × 9 high
Square size:   40mm
```

---

## ABT CODEBASE
```
ABT_ROOT = /Users/acalapai/PycharmProjects/Datalus/ABT_Software-main
ABT_3D_MODULE = /Users/acalapai/PycharmProjects/Datalus/ABT_Software-main/Modules_3D
```

YAML format must match exactly what ABT expects — always reference existing YAMLs in DatalusCalibration/yamls/ for format.

---

## CALIBRATION VIDEOS
```
SESSION_250707:
  102: CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4
  108: CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4
  113: CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4
  117: CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4

SESSION_250708:
  102: CalibrationVideos/250708/_2_102_20250708161657.mp4
  108: CalibrationVideos/250708/_2_108_20250708161657.mp4
  113: CalibrationVideos/250708/_2_113_20250708161657.mp4
  117: CalibrationVideos/250708/_2_117_20250708161657.mp4
```

---

## PAPER TARGET

Nature Communications — proof-of-concept methods paper. Central claim: first fully automated, markerless, continuously operating 3D behavioral tracking system for identified non-human primates in a naturalistic captive environment. Two animals (Elm and Jok), 40 hours of data, zero human supervision after setup. Key result needed: 3D trajectories of both animals across all 20 sessions.

---

## OUTPUT RULES

- HomeCage outputs → HomeCage_output/
- PriCaB outputs → PriCaB_output/
- Benchmark outputs → PoseModelBenchmark/
- Never write to DatalusCalibration/, Measurements/, ABT_Software-main/, CalibrationVideos/
- Always update datalus_config.json at the end of every script
- All viewers output as single self-contained HTML files
- All numpy outputs as .npz with descriptive field names