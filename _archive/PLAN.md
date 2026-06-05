================================================================================
HomeCage — Human Self-Calibration
Full Development Plan: Based on PriCaB v1.2 Successful Architecture
Prepared for: HomeCage_SelfCalibration_Human pipeline development
Author: A. Calapai (acalapai@dpz.eu) / Claude Sonnet 4.6
Date: 2026-03-30
================================================================================

NOTE: This plan is an exact copy of the PriCaB v1.2 architecture report,
adapted as the starting blueprint for the HomeCage human self-calibration
pipeline. The PriCaB pipeline succeeded (14.0px reprojection error, 12/16
cameras placed). The HomeCage adaptation will use the same architecture with
the following key differences:
  - 4 cameras (102, 108, 113, 117) instead of 16
  - Person-height scale recovery used as in PriCaB (PERSON_HEIGHT_MM = 1700)
  - CAMERA POSITIONS (T) ARE FIXED INPUTS — they are not solved by the pipeline.
    Physical positions in mm are known ground truth and hardcoded:
      102: [300, 3260, 540]  108: [1850, 0, 2480]
      113: [50, 0, 550]      117: [2080, 3070, 2550]
    The pipeline only solves for R (camera orientation/rotation).
  - ROOM SIZE IS FIXED — room_mm: x=2240, y=3400, z=3260. Used directly for
    the viewer room box. Not derived from camera positions.
  - Room alignment uses known positions for Procrustes rotation-only alignment
    (replaces PriCaB's SVD floor-leveling + yaw alignment). After alignment,
    T is set directly to the known physical position for each camera.
  - Two calibration sessions: 250707 and 250708 (see CLAUDE.md for paths)
  - Output: HomeCage_SelfCalibration_Human/output/ (poses, viewer, yamls)

Items marked [ADAPT] require changes from the PriCaB original.
Items with no annotation are carried over as-is.


────────────────────────────────────────────────────────────────────────────────
1. PROJECT CONTEXT
────────────────────────────────────────────────────────────────────────────────

[ADAPT] HomeCage human self-calibration is a marker-free, human-body-based
multi-camera calibration pipeline for the HomeCage 4-camera setup at the
Deutsches Primatenzentrum (DPZ), Göttingen. The goal is to calibrate the 4
cameras (102, 108, 113, 117) mounted in the primate home cage without the use
of any checkerboard or physical fiducials.

Instead, a human technician walks through the empty cage while all cameras
record simultaneously. The pipeline:
  1. Detects the person's 2D body pose in every frame
  2. Solves camera ORIENTATION (R) from cross-view keypoint correspondences
     (camera POSITIONS are given — they are not solved)
  3. Recovers metric scale from the known height of the person (1700mm)
  4. [ADAPT] Aligns the orientations to the physical room frame via Procrustes
     rotation, then sets each camera's T to its known physical position exactly
  5. Produces an interactive 3D HTML viewer and ABT-compatible YAML files

The pipeline feeds into ABT (Animal Behavior Tracker), the broader software
ecosystem for analyzing primate behavior using multi-camera 3D reconstruction.

Hardware:
  - 4 × Hikrobot MV-CH120-10GC cameras (Sony IMX304 sensor, 12 MP)
  - 8mm C-mount lenses
  - [ADAPT] Resolution: 2048 × 1496 px (HomeCage cameras, not 1500)
  - Frame rate during calibration: 5 fps

[ADAPT] Camera layout and known positions (from datalus_config.json):
  - 102: x=300,  y=3260, z=540   mm  (status: reference)
  - 108: x=1850, y=0,    z=2480  mm  (status: chained_via_113)
  - 113: x=50,   y=0,    z=550   mm  (status: direct)
  - 117: x=2080, y=3070, z=2550  mm  (status: direct)

Repository: /Users/acalapai/PycharmProjects/Datalus
[ADAPT] Main pipeline: HomeCage_SelfCalibration_Human/HomeCage_HumanCalib.py
Output: HomeCage_SelfCalibration_Human/output/ (not git-tracked except YAMLs)

[ADAPT] Input videos (two sessions, see CLAUDE.md):
  SESSION_250707:
    CAM_102 = CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4
    CAM_108 = CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4
    CAM_113 = CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4
    CAM_117 = CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4

  SESSION_250708:
    CAM_102 = CalibrationVideos/250708/_2_102_20250708161657.mp4
    CAM_108 = CalibrationVideos/250708/_2_108_20250708161657.mp4
    CAM_113 = CalibrationVideos/250708/_2_113_20250708161657.mp4
    CAM_117 = CalibrationVideos/250708/_2_117_20250708161657.mp4


────────────────────────────────────────────────────────────────────────────────
2. VERSION HISTORY (PriCaB reference — milestones to replicate)
────────────────────────────────────────────────────────────────────────────────

The PriCaB pipeline was developed iteratively. The HomeCage adaptation will
target the v1.2 stable feature set directly (no need to repeat the same
incremental journey). Key milestones from PriCaB for reference:

  v1.0  — Core calibration pipeline: inspect → extract → detect → intrinsics
           → extrinsics → YAMLs → viewer → config.
           Viewer was 2D only (Plotly scatter of camera positions + video strip).

  v1.1  — Correct aspect ratio, 5fps playback, keypoint overlay in viewer.

  v1.3  — Three.js 3D viewer with cylinder bones, CSS2D camera labels,
           OrbitControls for mouse navigation.

  v1.4  — Orbit pivot at scene centroid, transparent room walls with edge outlines.

  v1.5  — Floor leveling via SVD pitch/roll correction.

  v1.6  — X-axis mirror fix, floor offset, head centroid sphere.

  v1.7  — Camera strip layout, playback speed selector.

  v1.8  — Bone plausibility filter (700mm max neighbor distance).

  v1.9  — Robust triangulation: pairwise DLT + RANSAC 300mm inlier filter.

  ── v1.2 STABLE (PriCaB, frozen 2026-03-29) ─────────────────────────────────

  Commit 4b39e7e — info panel, acquisition script, wall snap, camera strip ordering
  Commit 8bf3f7d — recording, CoM ellipses, full info panel, acquisition script fixes
  Commit d7eb748 — YOLO26x-pose switch, semitransparent CoM sphere
  Commit 2f0b61d — fix screen recording (tag: v1.2-pricab-stable)

[ADAPT] HomeCage target: build v1.0 of HomeCage_HumanCalib.py implementing
the full v1.2 stable feature set from the start, adapted for 4 cameras with
known physical positions.


────────────────────────────────────────────────────────────────────────────────
3. FEATURES TO BUILD (based on PriCaB v1.2 stable)
────────────────────────────────────────────────────────────────────────────────

3.1  POSE MODEL: YOLO26x-pose
─────────────────────────────
Model:  yolo26x-pose.pt  (120 MB, Ultralytics v8.4.0)
Cached: /Users/acalapai/PycharmProjects/Datalus/yolo26x-pose.pt

Detection thresholds:
  - bbox confidence: 0.30
  - keypoint confidence: 0.50 (lower to 0.25 if few detections detected)
  - Sampling rate: 1 frame per 500ms (2 Hz)

Output: pose_<cam>.txt (one detection per frame, best-scoring human)
Format per detection (2 lines):
  Line 1: frame_idx  x1 y1 x2 y2  bbox_conf
  Line 2: kp0_x kp0_y kp0_conf ... kp16_x kp16_y kp16_conf

Skipped if pose_<cam>.txt already exists (resume logic).

[ADAPT NOTE: HomeCage cameras may capture two sessions merged. Consider
whether to process each session independently or concatenate. Simpler approach:
process each session separately and use the session with better coverage.]


3.2  CAMERA STRIP PERIMETER ORDERING
──────────────────────────────────────
Sort cameras by azimuthal angle around room centroid.
[ADAPT] With only 4 cameras this means a single-row strip is sufficient.

Implementation:
    cx_floor = sum(cam_centres[c][0] for c in cam_ids_s) / len(cam_ids_s)
    cy_floor = sum(cam_centres[c][1] for c in cam_ids_s) / len(cam_ids_s)
    def _cam_angle(c):
        dx = cam_centres[c][0] - cx_floor
        dy = cam_centres[c][1] - cy_floor
        return math.atan2(dy, dx)
    cam_ids_s = sorted(cam_ids_s, key=_cam_angle)


3.3  WALL SNAPPING (NO MARGIN)
────────────────────────────────
Room walls snapped exactly to camera bounding box:
  xMin, xMax = min/max of camera X positions
  yMin, yMax = min/max of camera Y positions
  zMax = max of camera Z positions + 400mm (ceiling clearance only)

[ADAPT] For HomeCage, can also snap to known room dimensions (2240 × 3400mm
footprint, 3260mm height) rather than camera extents, since room dimensions
are precisely known.


3.4  FULL INFO PANEL (7 SECTIONS)
───────────────────────────────────
Left panel sections:
  1. Recording:      Session name, fps, duration, resolution, camera count
  2. Pose Detection: Model, schema, sampling rate, confidence thresholds, frames
  3. Calibration:    Method, reference camera, reprojection error, scale factor
  4. 3D Reconstruction: Triangulation method, RANSAC threshold, bone gate, frames
  5. Cameras:        Per-camera table: ID | detection count | intrinsics source
  6. Live:           Frame index, Hip XYZ (mm), cameras detecting, visible KPs
  7. Attribution:    Lab: Cognitive Ethology Lab, DPZ Göttingen | acalapai@dpz.eu

Live section updated by setFrame() on every frame advance, slider scrub, playback.


3.5  CoM MARKER: SEMITRANSPARENT SPHERE
─────────────────────────────────────────
  const _comMesh = new THREE.Mesh(
      new THREE.SphereGeometry(220, 24, 16),
      new THREE.MeshPhongMaterial({
          color: 0x44aaff, emissive: 0x0a2244,
          opacity: 0.30, transparent: true,
          shininess: 80, depthWrite: false
      })
  );

Properties: radius=220mm, blue, 30% opacity, depthWrite=false to prevent
z-fighting with skeleton bones. Follows hip midpoint triangulation each frame.


3.6  SCREEN RECORDING (MediaRecorder API)
──────────────────────────────────────────
  - navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 30 }, audio: false })
  - NO preferCurrentTab (causes immediate stream termination)
  - MIME: try webm/vp9 → webm/vp8 → webm → mp4/h264 → mp4
  - Download only if recChunks.length > 0

Note: Chrome saves .webm (not macOS-native). Convert:
  ffmpeg -i recording.webm -c:v copy recording.mp4


3.7  VIEWER HEADER
────────────────────
Left:  "HomeCage — Human Self-Calibration / Multi-camera 3D Pose Viewer"
Right: "acalapai@dpz.eu | YYYY-MM-DD" (build date injected at generation time)


────────────────────────────────────────────────────────────────────────────────
4. CALIBRATION PIPELINE — TECHNICAL DEEP DIVE
────────────────────────────────────────────────────────────────────────────────

Full 8-stage pipeline (PriCaB v1.2 architecture, HomeCage adaptations noted).

STAGE 1 — INSPECT
  Scan the video folder(s). Auto-detect camera IDs from filenames.
  Read video metadata: resolution, fps, frame count, duration.
  [ADAPT] Handle two sessions (250707, 250708). Choose one or concatenate.
  Recommended: try 250708 first (single session, simpler). Fall back to
  concatenating both sessions if coverage is insufficient.

STAGE 2 — EXTRACT FRAMES
  Sample 1 frame per 500ms from each video (FRAME_INTERVAL_MS = 500).
  Output: output/frames/<cam_id>/<cam_id>_frame_NNNNNN.png
  Skipped if frame directories already exist (resume logic).

STAGE 3 — POSE DETECTION
  Run YOLO26x-pose on every extracted frame.
  Detection thresholds: bbox_conf ≥ 0.30, kp_conf ≥ 0.50
  [ADAPT] If average detection confidence < 0.35 per camera, lower kp_conf to 0.25.
  Output: output/pose_<cam_id>.txt
  Format per detection (2 lines):
    Line 1: frame_idx  x1 y1 x2 y2  bbox_conf
    Line 2: kp0_x kp0_y kp0_conf ... kp16_x kp16_y kp16_conf
  Skipped if pose_<cam>.txt already exists.

STAGE 4 — INTRINSICS
  Load per-camera K and distortion from DatalusCalibration/intrinsics.npz if available.
  Fallback: thin-lens estimate:
    LENS_FOCAL_MM   = 8.0     VS-Technology V0828-MPY2 C-mount, confirmed 8mm
    SENSOR_WIDTH_MM = 14.16   Sony IMX304 1.1" format (image format 14.16×10.37mm,
                              confirmed from V0828-MPY2 datasheet).
                              Cameras run at 2×2 binning: 4096×3000 → 2048×1496.
                              Sensor width is unchanged by binning.
                              DO NOT use 11.2mm — that is the 1" format, not 1.1".
                              Using 11.2mm gives fx=1463px (26% too high) and
                              produces ~900mm systematic position error in self-calib.
    fx = fy = focal_mm × image_width_px / sensor_width_mm  →  8.0 × 2048 / 14.16 = 1157px
    cx = image_width / 2, cy = image_height / 2
    dist_coeffs = zeros (distortion on 1.1" sensor is 0.60% per lens spec — negligible)

STAGE 5 — EXTRINSICS

  [ADAPT — KEY DIFFERENCE FROM PriCaB]
  HomeCage has known physical camera positions. The extrinsics pipeline should:

  5.1  Anchor pair selection (same as PriCaB)
       Count frames where both cameras simultaneously detect a full-body pose:
         - bbox_conf ≥ 0.70
         - SCALE_KPS = [0,5,6,11,12,15,16] (nose, shoulders, hips, ankles)
         - All SCALE_KPS visible with kp_conf ≥ 0.50
         - Vertical body span ≥ 30% of image height
       Pair with most co-detections = anchor pair.

  5.2  recoverPose on anchor pair (same as PriCaB)
       Build 2D-2D correspondences. Normalise by focal length.
       cv2.findEssentialMat (RANSAC) → cv2.recoverPose → (R, T unit scale).
       Set cam_A as origin (R=I, T=0), cam_B at (R_cb, T_cb_unit).

  5.3  Metric scale from person height (same as PriCaB)
       For each full-body frame in the anchor pair:
         - Triangulate head keypoint (nose, kp0) and ankle keypoints (15, 16)
         - Compute Euclidean 3D distance: head-to-ankle
       scale = PERSON_HEIGHT_MM / median(all head-to-ankle distances)
       PERSON_HEIGHT_MM = 1700  (measure actual walker height and update)
       Apply to T: T_metric = T_cb_unit × scale
       All distances are now in millimetres.

  5.4  Triangulate SCALE_KPS as metric 3D landmarks (same as PriCaB)
       Using metric anchor pair: triangulate SCALE_KPS from qualifying frames.
       Cheirality filter: keep only points with positive Z in both cameras.

  5.5  solvePnP for remaining cameras (same as PriCaB)
       Sort by overlap with already-placed cameras.
       For each unplaced camera: find shared 3D-2D landmarks → solvePnPRansac.
       [ADAPT] Only 2 remaining cameras (108, 113 or 117) after anchor pair.

  5.6  Bundle adjustment (same as PriCaB)
       scipy.optimize.least_squares, TRF solver, soft-L1 loss.
       Optimise: all non-anchor camera poses (Rodrigues + T).
       Fix: 3D landmarks (held constant).
       Report median reprojection error before and after BA.

  5.7  [ADAPT] Room alignment via Procrustes to known positions
       Replace PriCaB's SVD floor-leveling + yaw alignment with a single
       Procrustes step using known physical camera positions:
         - Known positions: {102: [300,3260,540], 108: [1850,0,2480],
                             113: [50,0,550], 117: [2080,3070,2550]} mm
         - Reconstructed metric positions: from solvePnP output
         - Procrustes alignment (no re-scaling — metric scale already set in 5.3):
             Compute centroids of both point sets.
             Subtract centroids. Solve SVD of (reconstructed^T @ known).
             R_align = V @ U^T  (rotation only, s=1 enforced)
             T_align = known_centroid − R_align @ reconstructed_centroid
         - Apply R_align, T_align to all camera poses and all 3D landmarks
         - Result: reconstruction is in the physical room coordinate system,
           with gravity and yaw already correct (encoded in known positions).

       Implementation: numpy SVD (no scipy needed for the rotation solve).
       NOTE: Do NOT use scipy.spatial.procrustes — it rescales, which would
       override the metric scale already recovered from person height.

  5.8  Chirality fix (same as PriCaB, may not be needed with Procrustes)
       After Procrustes, the reconstruction should already have correct handedness.
       Check: cam113 (x≈50mm, near left wall) should have lower X than cam117 (x≈2080mm).
       If mirrored: negate all X coordinates and flip X column of all R matrices.

STAGE 6 — WRITE YAMLs
  One YAML per camera under output/yamls/<cam_id>.yaml
  ABT convention: stores K^T and R^T; T as-is.
  OpenCV FileStorage format with !!opencv-matrix tags.
  All 4 cameras get a YAML (failed cameras get identity pose).

STAGE 7 — BUILD VIEWER
  Self-contained single HTML file (images base64 embedded).
  Open directly in browser — no server required.
  Technology: Three.js r0.160.0 (ES modules via importmap from CDN).
  [ADAPT] Use proxy videos (H.264 fast-start, 640px wide) for video strip.
  Full viewer specification in section 5 below.

STAGE 8 — UPDATE CONFIG
  Writes/updates datalus_config.json with:
  - Camera positions (3D XYZ in mm, post-Procrustes)
  - Which cameras were placed vs failed
  - Reprojection error (before and after BA)
  - Calibration metadata (session name, date, model used)


────────────────────────────────────────────────────────────────────────────────
5. INTERACTIVE 3D VIEWER — TECHNICAL DEEP DIVE
────────────────────────────────────────────────────────────────────────────────

5.1  Layout Structure
  Top bar:      Title, email (acalapai@dpz.eu), build date
  Left panel:   Info panel (240px fixed width, 7 sections, live-updating)
  Centre:       Three.js 3D scene (fills remaining width)
  Bottom strip: Camera feed strip (single row, 4 cameras, sorted by azimuth)
  Bottom bar:   Playback controls

5.2  3D Scene — Coordinate System
  Physical world:  X (left→right), Y (front→back depth), Z (up)
  Viewer space:    same X, same Y, Z up
  [ADAPT] Camera positions are already in physical mm from datalus_config.json.
  After Procrustes, all reconstructed positions are in this same frame.

  Conversion from OpenCV camera space (X right, Y down, Z forward):
    viewer_x = raw_x
    viewer_y = raw_z
    viewer_z = -raw_y

  CRITICAL: Three.js OrbitControls expects Y-up. Use camera.up.set(0,1,0)
  and map physical Z (height) to viewer Y, physical Y (depth) to viewer Z:
    R2T = (x, y_depth, z_height) => [x, z_height, y_depth]

  This avoids the coordinate system mismatch that broke the previous attempt.

5.3  3D Scene — Objects
  Camera markers:  Red octahedra (r=140mm), one per placed camera
                   CSS2D floating label shows camera ID
                   [ADAPT] Known positions provide ground truth for marker placement

  Skeleton:        COCO 17-point, CylinderGeometry bones (r=18mm)
                   Head keypoints 0-4 replaced by single sphere at centroid
                   Bones only drawn if both endpoints pass plausibility:
                     - Endpoint must have ≥1 COCO neighbor within 700mm

  CoM sphere:      Semitransparent blue sphere (r=220mm, opacity=30%)
                   Follows triangulated hip midpoint each frame
                   Always visible alongside skeleton

  Room walls:      6 transparent PlaneGeometry planes
                   [ADAPT] Snap to known room dimensions: 2240 × 3400 × 3260mm
                   OR snap to camera bounding box — either approach works

  Floor:           THREE.GridHelper at Z=0 (physical floor level)

  Lighting:        AmbientLight + DirectionalLight (from above)

5.4  3D Reconstruction (computed by Python, embedded in HTML as JSON)
  Hip midpoint:    All camera pairs triangulate the hip midpoint via DLT
                   RANSAC filter: discard results >300mm from median
                   Result: robust per-frame 3D hip position in room mm

  All 17 KPs:      Same procedure per keypoint, per frame

  Plausibility:    A keypoint rendered only if ≥1 COCO neighbor within 700mm

5.5  Playback Controls
  Speed selector:  0.4× / 0.6× / 1× / 1.5× / 2× (relative to 5Hz acquisition)
  Slider:          Scrub to any frame
  Bones button:    Toggle skeleton connector visibility
  Rec button:      MediaRecorder screen capture → download as .webm/.mp4

5.6  Camera Strip
  [ADAPT] Single row (4 cameras), sorted by azimuthal angle around room centroid
  Each cell: live video playback + YOLO bounding box overlay for current frame
  [ADAPT] Video source: proxy videos (H.264, fast-start, 640px wide)
    - Create with: ffmpeg -y -i src -vf scale=640:-2 -c:v libx264 -crf 28
                   -preset veryfast -an -movflags +faststart dst
  [ADAPT] Use <video> element directly (Safari-native approach, no canvas drawImage)
  [ADAPT] If combining two sessions, build proxy only from the session used in
  calibration, or build a combined proxy with correct frame offsets tracked.


────────────────────────────────────────────────────────────────────────────────
6. EXPECTED RESULTS
────────────────────────────────────────────────────────────────────────────────

[ADAPT] HomeCage-specific expectations:

  Cameras available:  4 (102, 108, 113, 117)
  Cameras expected to place: all 4 (known positions → Procrustes should always work)
  Expected reprojection error: < 14px (fewer cameras = better constraint per camera)

  Potential failure modes:
  - Insufficient human visibility if cameras are aimed at cage interior (monkeys)
  - Two-session mismatch if sessions are combined incorrectly
  - HomeCage camera angles may differ significantly, reducing keypoint overlap
  - Low-mounted cameras (102 z=540mm, 113 z=550mm) may miss the head/full-body
    detection required for person-height scale recovery

  Success criteria:
  - All 4 cameras placed
  - Reprojection error < 20px (acceptable) / < 14px (good) / < 8px (excellent)
  - Camera positions within ±50mm of known physical positions
  - Person trajectory visible and plausible in 3D viewer


────────────────────────────────────────────────────────────────────────────────
7. KNOWN LIMITATIONS FROM PriCaB (carry forward)
────────────────────────────────────────────────────────────────────────────────

7.1  Person height assumption (same as PriCaB)
  Scale is derived by assuming the calibration walker is exactly 1700mm tall.
  A 1% error in person height = 1% scale error across all 3D positions.

  Mitigation: measure the actual walker's height before each calibration session
  and update PERSON_HEIGHT_MM accordingly.

  [ADAPT] The Procrustes room alignment in Stage 5.7 absorbs residual scale
  errors introduced by the height assumption — camera positions in the aligned
  frame will match the known physical positions regardless. However, the
  triangulated animal trajectories will still carry the height-induced scale
  error, so measuring actual person height remains important.

7.2  Reprojection error
  PriCaB achieved 14.0px with 12/16 cameras and estimated intrinsics.
  HomeCage should do better: fewer cameras, known positions, identical hardware.
  Target: < 10px.

7.3  Temporal sync in freerun mode
  At 5fps, sync error ≈ ±100ms. Frame index used as proxy for time.
  Calibration correspondence based on frame index may have small errors.
  Acceptable for calibration purposes.

7.4  Screen recording saves as .webm (Chrome)
  Chrome's MediaRecorder produces .webm files which macOS cannot natively play.
  Convert: ffmpeg -i recording.webm -c:v copy recording.mp4

7.5  [NEW IN HOMECAGE] Multi-session frame offset
  If sessions 250707 and 250708 are concatenated, proxy video for one session
  may not cover the other. Track VIDEO_OFFSET = first_frame_of_session_in_combined
  and subtract when seeking: video.currentTime = (frame - VIDEO_OFFSET) / FPS.

7.6  [NEW IN HOMECAGE] Camera sightlines
  HomeCage cameras (102, 113) are at low height (z=540mm, 550mm) — they may
  capture the person's lower body more than face/head. Expect fewer full-body
  detections from low cameras. Anchor pair selection should handle this by
  picking the pair with most co-detections (likely 108+117, both at ~2500mm height).


────────────────────────────────────────────────────────────────────────────────
8. FILE STRUCTURE (target)
────────────────────────────────────────────────────────────────────────────────

/Users/acalapai/PycharmProjects/Datalus/HomeCage_SelfCalibration_Human/
│
├── HomeCage_HumanCalib.py        Main calibration pipeline
├── PLAN.md                       This file
│
└── output/                       (not git-tracked except YAMLs)
    ├── yamls/                    Per-camera YAML files (git-tracked)
    │   └── <cam_id>.yaml         ABT-compatible calibration
    ├── frames/                   (not git-tracked)
    │   └── <cam_id>/             Extracted PNG frames
    ├── pose_<cam_id>.txt         YOLO detections (not git-tracked)
    ├── poses.npz                 Raw camera poses
    ├── poses_scaled.npz          Metric camera poses (post-Procrustes)
    ├── proxy/                    H.264 fast-start proxy videos
    │   └── proxy_<cam_id>.mp4
    └── homecage_human_viewer.html   Interactive viewer (not git-tracked)


────────────────────────────────────────────────────────────────────────────────
9. DEPENDENCIES
────────────────────────────────────────────────────────────────────────────────

Python:
    opencv-python >= 4.8
    numpy >= 1.24
    scipy >= 1.11     (for Procrustes alignment: scipy.spatial.procrustes)
    ultralytics >= 8.0  (for YOLO26x-pose, downloads model on first run)

JavaScript (CDN, no install):
    three@0.160.0  — Three.js core
    OrbitControls  — mouse navigation
    CSS2DRenderer  — floating camera labels

External:
    ffmpeg  — for creating H.264 proxy videos (Homebrew: /opt/homebrew/bin/ffmpeg)


────────────────────────────────────────────────────────────────────────────────
10. IMPLEMENTATION ORDER
────────────────────────────────────────────────────────────────────────────────

Recommended build sequence (one working stage before the next):

  Step 1  — Stage 1-3: Inspect, extract frames, run YOLO26x-pose on 250708 session.
             Verify: pose_<cam>.txt files created, detection counts reasonable.

  Step 2  — Stage 4-5: Intrinsics + full extrinsics pipeline.
             Verify: all 4 cameras placed, reprojection error < 20px.

  Step 3  — Stage 6: Write YAMLs. Verify ABT can load them.

  Step 4  — Create proxy videos for 250708 session.

  Step 5  — Stage 7: Build viewer (minimal first: 3D scene + camera markers only).
             Verify: cameras at correct physical positions, room walls correct.

  Step 6  — Add 3D skeleton + CoM sphere to viewer.
             Verify: person trajectory is physically plausible.

  Step 7  — Add camera strip + video playback.
             Verify: videos play, bounding boxes overlay correctly.

  Step 8  — Add full info panel, screen recording, polish.

  Step 9  — Stage 8: Update datalus_config.json.

  Step 10 — Freeze as HomeCage_SelfCalibration_Human v1.0.


────────────────────────────────────────────────────────────────────────────────
11. REFERENCE: PriCaB v1.2 GIT TAG
────────────────────────────────────────────────────────────────────────────────

Source of this plan:
  Tag:    v1.2-pricab-stable
  Commit: 2f0b61d0b910668714ee49955b8968e1c1bb33c8
  Script: PriCaB_SelfCalibration_Human/PriCaB_HumanCalib.py
  Report: PriCaB_SelfCalibration_Human/pricab_v1_2_report.txt


================================================================================
END OF PLAN — HomeCage_SelfCalibration_Human v0.1 (blueprint)
================================================================================
