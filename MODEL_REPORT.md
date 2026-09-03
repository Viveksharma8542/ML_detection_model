# Pothole Alert System — Full Performance Report

**System:** ESP32-CAM live stream → PC (Python + YOLO) → UDP → ESP8266 → ISD1820 Voice Alert
**Model:** `best (1).pt` (YOLO11n — Nano variant)
**Input Size:** 320 × 320 px | **Confidence Threshold:** 0.65
**Test Hardware:** CPU-only (torch 2.13.0+cpu, ultralytics 8.4.95)
**Test Method:** 100 timed inference runs after 10 warm-ups, 480×640 dummy frame

---

## System Architecture

```
┌─────────────────────┐      MJPEG stream        ┌──────────────────────┐
│  ESP32-CAM          │ ────────────────────────► │  PC / Laptop         │
│  (camera + stream)  │   http://172.16.2.3/stream│  Python + YOLO11n    │
└─────────────────────┘                           │  detection            │
                                                  └──────────┬───────────┘
                                                             │ UDP packet
                                                             │ "POTHOLE"
                                                  ┌──────────▼───────────┐
                                                  │  ESP8266 (receiver)  │
                                                  │  GPIO5 pulse 150 ms  │
                                                  └──────────┬───────────┘
                                                             ▼
                                                  ┌─────────────────────┐
                                                  │  ISD1820 + Speaker  │
                                                  │  "Pothole ahead!"   │
                                                  └─────────────────────┘
```

| Component | Role |
|---|---|
| ESP32-CAM (AI-Thinker) | Captures video, serves MJPEG stream on port 80 |
| PC Python script | Reads stream, runs YOLO detection, sends UDP alerts |
| ESP8266 | Listens on UDP port 4210, pulses GPIO5 on "POTHOLE" |
| ISD1820 + Speaker | Plays recorded voice message when PLAYE pin triggered |

---

## 1. Camera & Streaming (ESP32-CAM)

**Settings used in firmware:**
| Setting | Value |
|---|---|
| Board | AI-Thinker ESP32-CAM |
| Frame size | `FRAMESIZE_UXGA` (see warning below) |
| JPEG quality | 10–12 |
| PSRAM | Present (frames in PSRAM, fb_count = 2) |
| XCLK | 20 MHz |
| Wi-Fi sleep | Disabled (lower latency) |
| Grab mode | `CAMERA_GRAB_WHEN_EMPTY` (always newest frame) |
| Stream type | MJPEG (multipart/x-mixed-replace), ~20 FPS header |

> **⚠️ WARNING — check your code:** `FRAMESIZE_UXGA` is **1600×1200**, not 320×240 (that's QVGA). Your comment says "320x240" but the constant is UXGA. If the camera truly runs at UXGA, the frames are huge (100–300 KB each) → stream FPS will be very low and the Python side will decode slowly. For your 320×320 YOLO input, use `FRAMESIZE_QVGA` (320×240) — it will give 3–5× higher FPS with no accuracy loss at imgsz=320.

---

## 2. Model Memory Consumption

| Metric | Value |
|---|---|
| Model file size on disk | 5.21 MB |
| Total model parameters | 2,590,035 (≈2.59 M) |
| Model layers | 101 |
| Python-side allocation per inference | 0.61 MB (negligible, frames reused) |
| Total process RAM (model + torch runtime) | ≈368 MB |

**Note:** The ~368 MB is the full Python + PyTorch + OpenCV runtime. The model weights themselves need only ~5 MB. The ESP32-CAM and ESP8266 use zero extra RAM for detection — all heavy work is on the PC.

## 3. Latency (Minimum Time)

| Latency Metric | Value |
|---|---|
| **Minimum inference latency** | **121.9 ms** |
| Average (mean) | 141.4 ms |
| Median (p50) | 134.6 ms |
| 95th percentile (p95) | 248.3 ms |
| Worst case (max) | 289.1 ms |

**Minimum = 121.9 ms** → even in the best case, one frame takes ~122 ms on this CPU.

## 4. Inference Time

- **Per-frame inference:** 121.9 – 289.1 ms (avg ≈ 141.4 ms) — this is the `model.predict()` call only
- In your pipeline, the reader thread fetches frames (READ time) in parallel with inference (INF time), so real display FPS is higher than raw numbers

## 5. Operation Speed (Giga-FLOPS)

| Metric | Value |
|---|---|
| Compute cost | **6.31 GFLOPs** per inference |
| Model class | YOLO11n — smallest & fastest YOLO11 variant |
| Effective compute rate (CPU) | ≈ 44.6 GFLOPS (6.31 GFLOPs ÷ 141.4 ms) |

## 6. Throughput

| Metric | Value |
|---|---|
| Frames per second | **7.1 FPS** |
| Inferences per minute | ≈ 426 |
| Theoretical max (at min latency) | ≈ 8.2 FPS |

## 7. Model Parameters & Delivery

| Metric | Value |
|---|---|
| Parameters (fused) | 2,582,347 |
| GFLOPs | 6.31 |
| File size | 5.21 MB |
| **Delivery rate (effective)** | **7.1 frames/sec** (~141 ms/frame) |
| Classes | 1 (Pothole), multiple objects per frame |
| Confidence filter | 0.65 |

---

## 8. End-to-End System Latency (Pothole → Speaker)

```
[ESP32-CAM capture + JPEG encode]   ~40–80 ms
[Wi-Fi transfer of frame]           ~5–20 ms
[Python read + decode]              ~10–30 ms
[YOLO inference]                    ~122–141 ms  (avg)
[UDP send to ESP8266]               <1 ms
[ESP8266 GPIO pulse → ISD1820]      starts immediately (pulse 150 ms)
─────────────────────────────────────────────
TOTAL (pothole in frame → voice)   ≈ 180–270 ms
```

**During a bike ride (25–40 km/h = 7–11 m/s):**
- Distance covered in ~270 ms: **1.9–3 m** — the voice alert fires before you reach the pothole
- At ~7 FPS you get **4–7 frames** of each pothole (visible ~15–30 m ahead)
- UDP cooldown (5 s) prevents spam for long stretches of multiple potholes

**Verdict for bike use:** The system is **good for a warning buzzer** at city/average speed (≤ 40–50 km/h). It is NOT good for high-speed riding or precision braking — for that you'd need GPU/edge acceleration (30+ FPS).

---

## Recommendations

1. **Fix frame size bug:** change `FRAMESIZE_UXGA` → `FRAMESIZE_QVGA` (320×240). Faster stream, smaller frames, no accuracy loss at imgsz=320.
2. **Lower conf to 0.5–0.6** — small or far potholes get missed at 0.65, and at 7 FPS you can't afford misses.
3. **GPU / TensorRT** → 30–100× faster (60–200 FPS). Use if laptop has any NVIDIA GPU.
4. **ONNX export** (`model.export(format="onnx")`) → 1.5–3× CPU speedup without GPU.
5. **imgsz=256** → ~40% fewer FLOPs, faster latency, minor accuracy loss.
6. Keep `grab_mode = CAMERA_GRAB_WHEN_EMPTY` + `fb_count = 2` — already correct for low lag.

---

## Benchmark Method (reproducible)

```python
import time, numpy as np
from ultralytics import YOLO
model = YOLO("best (1).pt")
frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
for _ in range(10):                       # warm-up
    model.predict(source=frame, imgsz=320, conf=0.65, verbose=False)
for _ in range(100):                      # timed runs
    t0 = time.perf_counter()
    model.predict(source=frame, imgsz=320, conf=0.65, verbose=False)
    print((time.perf_counter() - t0) * 1000)
```

| Statistic | Value |
|---|---|
| min | 121.9 ms |
| avg | 141.4 ms |
| p50 | 134.6 ms |
| p95 | 248.3 ms |
| max | 289.1 ms |
| Throughput | 7.1 FPS |

---

*Generated by automated benchmark of `best (1).pt` — benchmark_model.py (F:\coding\opencv)*
