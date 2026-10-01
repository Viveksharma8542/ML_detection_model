# Pothole Detector — Android App

Live pothole detection on phone camera. Same YOLO11n model as `try(1)` (`best.onnx`,
160x160, 1 class), running on-device with ONNX Runtime — no ESP32 / WiFi / laptop needed.

## What's inside

| File | Purpose |
|---|---|
| `app/src/main/assets/best.onnx` | YOLO11n pothole model (9.96 MB, input `[1,3,160,160]`, output `[1,5,525]`), copied from `try(1)` |
| `.../Detector.kt` | ORT session, letterbox preprocess (160), conf 0.5, NMS IoU 0.45, max 10 |
| `.../OverlayView.kt` | Green boxes + `Pothole 0.xx` labels over preview |
| `.../MainActivity.kt` | CameraX live feed + analyzer + FPS/inference-ms overlay |
| `app/build.gradle` | CameraX BOM 1.3.4 + `onnxruntime-android:1.20.0` |

Package `com.pothole.detector`, minSdk 26.

## Build & install

1. Android Studio → **Open** → select `PotholeApp` folder → let Gradle sync (downloads CameraX + ORT).
2. Phone: enable **Developer options → USB debugging**, connect via USB.
3. Press **Run** (green triangle) → app installs, grant Camera permission.
4. Point at road — boxes appear with `INF: xxms FPS: xx`.

## Notes

- Phone GPU/NNAPI typically gives 10–30 ms inference at 160 input.
- `CONF_THRESHOLD = 0.5` in `Detector.kt` (laptop used 0.65; 0.5 shows more on-device).
- Threshold/NMS values mirror `try(1)/detect.py` (`CONF 0.65→0.5`, `MAX_DET 10`).
- Each box shows `Pothole 0.xx` + distance in meters (ground-plane model).
- Stats line shows `INF / FPS / POTHOLE! Xm / GPS lat,lon` (needs location permission + sky view).

## Distance calibration (2 minutes, do once per mount)

`Detector.kt` uses `DIST_SLOPE = -23.3 / DIST_INTERCEPT = 24.0` (phone ~1.2 m high,
slight downward tilt). For your mount:

1. Park bike, place markers at 3 m and 8 m ahead.
2. Note each marker's normalized box-bottom-y (0 = image top, 1 = bottom).
3. Solve: `SLOPE = (3-8)/(y3-y8)`, `INTERCEPT = 3 - SLOPE*y3`, update constants.
4. Rebuild — distances now read in meters for your setup.
