# Datalus — Project Configuration for Claude

## Calibration Pipeline Inputs

### Calibration Videos (2 sessions, 4 cameras each)
```
SESSION_250707:
  CAM_102 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_102_20250707154928.mp4"
  CAM_108 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_108_20250707154928.mp4"
  CAM_113 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_113_20250707154928.mp4"
  CAM_117 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250707/Calibration_4_117_20250707154928.mp4"

SESSION_250708:
  CAM_102 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_102_20250708161657.mp4"
  CAM_108 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_108_20250708161657.mp4"
  CAM_113 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_113_20250708161657.mp4"
  CAM_117 = "/Users/acalapai/ownCloud/Shared/HomeCage/CalibrationVideos/250708/_2_117_20250708161657.mp4"
```

### ABT Codebase
```
ABT_ROOT = "/Users/acalapai/ownCloud/Shared/HomeCage/ABT_Software-main"                  # root directory of the ABT repo
ABT_3D_TRANSFORM_MODULE = "/Users/acalapai/ownCloud/Shared/HomeCage/ABT_Software-main/Modules_3D"   # path to the 3D Transformation module file(s)
```

### Example YAML File
```
EXAMPLE_YAML = ""              # path to an existing camera calibration YAML consumed by ABT
```

### World Registration CSV
```
WORLD_REGISTRATION_CSV = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"    # path to CSV with columns: name, colmap_x, colmap_y, colmap_z, real_x_mm, real_y_mm, real_z_mm
```

### Output Directory
```
CALIBRATION_OUTPUT_DIR = "/Users/acalapai/ownCloud/Shared/HomeCage/DatalusCalibration"    # where to write the per-camera YAML files
```

## Checkerboard Specs
```
CHESSBOARD_INNER_CORNERS_W = 13   # number of inner corners along width  (e.g. 13)
CHESSBOARD_INNER_CORNERS_H = 9  # number of inner corners along height (e.g. 9)
CHESSBOARD_SQUARE_SIZE_MM  = 40  # physical square size in mm           (e.g. 25)
```

## Camera Specs (Hikrobot MV-CH120-10GC)
```
SENSOR_WIDTH_MM  = 11.2        # Sony IMX304 1.1" sensor width
FOCAL_LENGTH_MM  = 8.0         # C-mount lens focal length
IMAGE_WIDTH_PX   =             # fill after first run (expected: 4096)
IMAGE_HEIGHT_PX  =             # fill after first run (expected: 3000)
```

## Camera IDs
```
CAMERA_IDS = ["102", "108", "113", "117"]
```

## Notes
- Fill in all empty string values before running the calibration pipeline.
- The example YAML format drives the export step — ABT will not need changes.
- COLMAP assumed installed via Homebrew: /opt/homebrew/bin/colmap

## Claude Behavior
- do not ask to apply changes, apply changes directly
- Apply all code changes directly without asking for approval.
- Never prompt the user to confirm before editing or creating files.