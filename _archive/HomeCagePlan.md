Read CLAUDE.md, PLAN.md, and every .txt, .md, and .json file at the project root before doing anything else. Pay particular attention to pricab_v1_2_report.txt — it documents the full technical approach that works and must be carried forward. Then build the HomeCage calibration pipeline as described below. Do not ask for permission at any step. Do not stop between stages. Report only when fully complete.

CONTEXT — READ THIS CAREFULLY
The PriCaB pipeline (PriCaB_HumanCalib.py, v1.2-pricab-stable) is the working reference implementation. The HomeCage pipeline must follow the same architecture but adapt it for four key differences:

Only 4 cameras (102, 108, 113, 117) instead of 16
Camera positions and room dimensions are KNOWN — use them for alignment and validation instead of deriving from data
The subject is ANIMALS (two rhesus macaques: Elm and Jok), not a human — use the custom pose model yolo26m-pose_ElmJok.pt which is already in the repository
Scale is derived from known camera positions, not person height

Known geometry from datalus_config.json:

Room: x=2240, y=3400, z=3260 mm
Cam102: (300, 3260, 540) mm
Cam108: (1850, 3240, 2480) mm
Cam113: (50, 0, 550) mm
Cam117: (2080, 3070, 2550) mm

TASK
Write a single script HomeCage_Calibration.py. It must run end to end from raw videos to interactive HTML viewer with no manual steps.

STAGE 1 — Scan repository and inputs
Before writing any code, scan the entire repository:

Read all .txt, .md, .json, .yaml files at project root and in subdirectories
List all video files in Measurements/ and identify HomeCage sessions
List all files in DatalusCalibration/ — note which cameras have calibrated intrinsics in intrinsics.npz
Confirm yolo26m-pose_ElmJok.pt exists in the repository
Read datalus_config.json for camera positions and room dimensions
Print a full inventory before proceeding

STAGE 2 — Extract frames
Extract every frame from each of the 4 HomeCage camera videos at native FPS (read from metadata — never assume). Save to HomeCage_output/frames/{cam_id}/{cam_id}frame{index:06d}.png. Skip if already done. Store actual FPS per camera in config.
STAGE 3 — Animal pose detection
Run yolo26m-pose_ElmJok.pt on all extracted frames from all 4 cameras. This model is trained to detect Elm and Jok specifically — it outputs COCO 17-point keypoints per animal per frame. Detection thresholds: bbox_conf ≥ 0.30, kp_conf ≥ 0.50. Save as HomeCage_output/pose_{cam_id}.txt in the same format as PriCaB (one detection per animal per frame: frame_idx, bbox, 17 keypoints). Run cameras in parallel. Skip if already done. Report detection rate per camera and per animal.
STAGE 4 — Intrinsics
Load per-camera K and distortion from DatalusCalibration/intrinsics.npz for cameras 102, 108, 113, 117. If any camera is missing from intrinsics.npz, fall back to thin-lens estimate: fx = fy = (8.0/11.2) × image_width_px, cx = width/2, cy = height/2, zero distortion. Report which cameras used calibrated vs estimated intrinsics.
STAGE 5 — Extrinsics
Follow the same architecture as PriCaB v1.2 Stage 5 (documented in pricab_v1_2_report.txt section 4, Stage 5) with the following adaptations:
5.1 Anchor pair selection: use same co-detection counting approach. With only 4 cameras, all 6 pairs will likely have good overlap — pick the pair with the most high-confidence simultaneous animal detections. Use BOTH animals (Elm and Jok) as correspondences — double the keypoint pairs available.
5.2 recoverPose on anchor pair: same as PriCaB. Use RANSAC essential matrix from all 17 keypoints of both animals combined. Set cam102 as reference (R=I, T=0) if it is in the anchor pair, otherwise set whichever anchor camera is cam102 after solving.
5.3 Metric scale from known camera positions: DO NOT use person height. Instead compute scale from the known physical baseline between the anchor pair cameras. The baseline in mm is the Euclidean distance between their known positions from datalus_config.json. Divide by the recovered unit-scale translation magnitude to get the scale factor. Apply to all T vectors. Report scale factor and verify it produces physically plausible camera separations.
5.4 Triangulate landmarks: same as PriCaB — triangulate high-confidence keypoints from both animals as 3D world points using the now-metric anchor pair.
5.5 solvePnP for remaining cameras: same as PriCaB — use cv2.solvePnPRansac against the triangulated 3D landmarks. With only 4 cameras total this should be fast. Verify each solved position against known physical position from datalus_config.json — deviation should be < 400mm. Print deviation per camera.
5.6 Bundle adjustment: same as PriCaB — scipy TRF solver, soft-L1 loss, fix cam102, optimize remaining 3 cameras. Report median reprojection error before and after.
5.7 Room alignment using KNOWN geometry: this replaces the azimuth-based alignment used in PriCaB. Since camera positions are known, compute the rigid transformation that maps the reconstructed camera positions to the known physical positions using Procrustes alignment (scipy.spatial.transform.Rotation.align_vectors or equivalent). Apply this transformation to all camera centers and R matrices, recompute T = -R @ C. After alignment, print a validation table: recovered vs known position per camera, deviation in mm.
5.8 Chirality fix: same logic as PriCaB — verify cam113 (y=0 wall) is on the correct side relative to cam102 (y=3260 wall).
STAGE 6 — Write YAMLs
Same as PriCaB v1.2 — write HomeCage_output/yamls/{cam_id}.yaml in ABT-compatible format. ABT convention: store K^T and R^T, T as-is. OpenCV FileStorage format.
STAGE 7 — Triangulate animal trajectories
For each frame, triangulate the hip midpoint (mean of keypoints 11 and 12) for Elm and Jok separately using pairwise DLT from all camera pairs that detected that animal. Apply RANSAC filter: discard results > 300mm from median. Apply 5-frame rolling median. Filter to positions within room bounds (0–2240 X, 0–3400 Y, 0–3260 Z). Save to HomeCage_output/trajectories_Elm.npz and HomeCage_output/trajectories_Jok.npz with fields: frame_index, x_mm, y_mm, z_mm, confidence. Report frames triangulated per animal and success rate.
STAGE 8 — Build interactive viewer
Build HomeCage_output/homecage_viewer.html following the PriCaB v1.2 viewer architecture (pricab_v1_2_report.txt section 5) with these adaptations:
Layout:

added later:
Combine Stages 2 and 3 into a single stage. 
Do not extract frames to disk. Instead run yolo26m-pose_ElmJok.pt directly on each video file using the Ultralytics streaming API: model(video_path, stream=True). For each frame yielded by the stream, extract frame index, bounding boxes, and all 17 COCO keypoints with confidence scores for each detected animal. Save results to HomeCage_output/pose_{cam_id}.txt in the same format as before. Run all 4 cameras in parallel using multiprocessing. Skip cameras whose pose file already exists. Still read FPS and resolution from video metadata using cv2.VideoCapture before inference. 
Report detection rate per camera and per animal after all cameras complete.

Top bar: "HomeCage — Automated Behavioral Tracking / Multi-camera 3D Pose Viewer" + acalapai@dpz.eu + build date
Left: info panel (same 7-section structure as PriCaB v1.2)
Centre: Three.js 3D scene
Bottom: 2×2 camera strip (4 cameras in a single row or 2×2 grid)
Bottom bar: playback controls

Three.js scene:

Room wireframe box: exact dimensions 2240×3400×3260mm, drawn as EdgesGeometry
4 camera markers: red octahedra at KNOWN physical positions (not reconstructed — use ground truth from datalus_config.json for display)
Elm: blue skeleton + semitransparent blue CoM sphere (r=220mm, opacity=30%)
Jok: orange skeleton + semitransparent orange CoM sphere (r=220mm, opacity=30%)
Both animals tracked simultaneously, different colors
Skeleton: COCO 17-point, CylinderGeometry bones (r=18mm), same plausibility filter as PriCaB (700mm max bone length, ≥1 neighbor within 700mm)
Head keypoints 0–4 replaced by single sphere at centroid per animal
Floor grid at Z=0 (actual floor, since room dimensions are known)

Info panel sections:

Recording — session name, FPS, resolution, 4 cameras
Pose Detection — yolo26m-pose_ElmJok.pt, animals: Elm and Jok
Calibration — known positions used, solvePnP + BA, reprojection error
3D Reconstruction — pairwise DLT, RANSAC 300mm, bone gate 700mm
Cameras — per-camera table with known vs reconstructed position, deviation mm
Live — frame index, Elm XYZ, Jok XYZ, inter-animal distance mm (live per frame)
Attribution — Cognitive Ethology Lab, DPZ Göttingen, acalapai@dpz.eu

Playback controls:

Single slider controlling both camera strip and 3D scene
Play/Pause
Speed: 0.4×, 0.6×, 1×, 1.5×, 2× (base = actual video FPS)
Bones toggle
CoM toggle (hide/show CoM spheres independently from skeleton)
Rec button: MediaRecorder screen capture → download as .webm (same implementation as PriCaB v1.2 — without preferCurrentTab)
Frame sync offset slider: global -10 to +10 frames

Camera strip:

2×2 grid showing all 4 cameras simultaneously
Each tile: live video frame + bounding boxes for Elm (blue) and Jok (orange)
COCO skeleton overlay per animal
Camera ID label in corner
Preserve original aspect ratio via object-fit: contain

STAGE 9 — Update config
Add to datalus_config.json under homecage_calibration:
json{
  "approach": "solvePnP from animal keypoints + known positions + Procrustes alignment + BA",
  "model": "yolo26m-pose_ElmJok.pt",
  "animals": ["Elm", "Jok"],
  "reference_camera": "102",
  "reprojection_error_before_BA_px": "<value>",
  "reprojection_error_after_BA_px": "<value>",
  "per_camera_position_deviation_mm": {
    "102": 0,
    "108": "<value>",
    "113": "<value>",
    "117": "<value>"
  },
  "elm_frames_triangulated": "<n>",
  "jok_frames_triangulated": "<n>",
  "elm_success_rate_pct": "<value>",
  "jok_success_rate_pct": "<value>",
  "status": "DONE",
  "timestamp": "<now>"
}
REQUIREMENTS

Single script HomeCage_Calibration.py, no manual steps
All output to HomeCage_output/ — never modify DatalusCalibration/ or Measurements/
Skip stages whose output already exists
Print clear stage header for each stage
Never ask for permission for anything
Final three-line summary: reprojection error, frames triangulated per animal, viewer path

Run all 9 stages end to end. Report only when fully complete.