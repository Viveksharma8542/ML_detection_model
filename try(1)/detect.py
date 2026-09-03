import cv2
import os
import time
import threading
import socket
import select
import numpy as np
import json
import tracemalloc
import psutil
from ultralytics import YOLO


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "best (1).onnx"  # run export_onnx.py first; falls back to .pt
STREAM_URL = "http://172.16.6.78/stream"
ESP32_IP = "172.16.10.109"

FORCE_RAW_MODE = False
CONF_THRESHOLD = 0.65
INFER_SIZE = 160
MAX_DET = 10

ESP8266_IP = "172.16.10.80"
UDP_PORT = 4210
UDP_COOLDOWN = 5.0


# ============================================================
# BENCH MONITOR — queries ESP32 /bench for memory stats
# ============================================================

class BenchMonitor:

    def __init__(self, esp32_ip):
        self.esp32_ip = esp32_ip
        self.stats = {}
        for attempt in range(1, 4):
            print(f"[BENCH] Attempt {attempt}/3 — querying ESP32...")
            self._fetch()
            if self.stats:
                break
            time.sleep(2)

    def _fetch(self):
        try:
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )
            sock.settimeout(5)
            sock.connect((self.esp32_ip, 80))

            req = (
                f"GET /bench HTTP/1.1\r\n"
                f"Host: {self.esp32_ip}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            sock.sendall(req.encode())

            data = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                data += chunk
            sock.close()

            body_start = data.find(b"\r\n\r\n")
            if body_start == -1:
                return
            body = data[body_start + 4:]
            self.stats = json.loads(body.decode().strip())

        except Exception as e:
            print(f"[BENCH] ESP32 unreachable: {e}")

    def is_raw(self):
        return self.stats.get("camera_format") == "RGB565"

    def print_stats(self):
        if not self.stats:
            print("[BENCH] No ESP32 data")
            return
        heap = self.stats.get("heap_free_bytes", 0)
        psram = self.stats.get("psram_free_bytes", 0)
        raw_bytes = self.stats.get("raw_frame_bytes", 0)
        fmt = self.stats.get("camera_format", "?")
        res = self.stats.get("resolution", "?")
        print(f"  Format     : {fmt}")
        print(f"  Resolution : {res}")
        print(f"  Heap Free  : {heap / 1024:.1f} KB")
        print(f"  PSRAM Free : {psram / 1024:.1f} KB")
        print(f"  Raw Frame  : {raw_bytes / 1024:.1f} KB")


# ============================================================
# RAW FRAME READER — captures RGB565 directly from /capture
# No JPEG encode on ESP32, no JPEG decode on Python
# ============================================================

class RawFrameReader:

    def __init__(self, esp32_ip):
        self.esp32_ip = esp32_ip

    def capture(self):
        """Capture one raw RGB565 frame.
        Returns numpy BGR array or None.
        Opens a fresh connection each time — most reliable."""

        sock = None
        try:
            sock = socket.socket(
                socket.AF_INET,
                socket.SOCK_STREAM
            )
            sock.settimeout(30)
            sock.connect((self.esp32_ip, 80))

            req = (
                f"GET /capture HTTP/1.1\r\n"
                f"Host: {self.esp32_ip}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            )
            sock.sendall(req.encode())

            # Read headers + keep any body bytes in same chunks
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = sock.recv(8192)
                if not chunk:
                    raise ConnectionError("Header read failed")
                buf += chunk

            sep = buf.find(b"\r\n\r\n")
            header_str = buf[:sep].decode(errors="ignore")
            body = bytearray(buf[sep + 4:])

            content_length = 0
            width = 160
            height = 120
            chunked = False

            for line in header_str.split("\r\n"):
                lower = line.lower().strip()
                if lower.startswith("transfer-encoding:") and "chunked" in lower:
                    chunked = True
                elif lower.startswith("content-length:"):
                    content_length = int(
                        line.split(":", 1)[1].strip()
                    )
                elif lower.startswith("x-frame-width:"):
                    width = int(
                        line.split(":", 1)[1].strip()
                    )
                elif lower.startswith("x-frame-height:"):
                    height = int(
                        line.split(":", 1)[1].strip()
                    )

            print(
                f"[RAW] Headers: CL={content_length} "
                f"W={width} H={height} chunked={chunked}"
            )

            # ---- Read body: handles chunked, CL, or EOF modes ----
            def dechunk(src):
                out = bytearray()
                i = 0
                while True:
                    j = src.find(b"\r\n", i)
                    if j == -1:
                        return None
                    try:
                        size = int(src[i:j].split(b";")[0], 16)
                    except ValueError:
                        return None
                    i = j + 2
                    if size == 0:
                        return bytes(out)
                    end = i + size + 2
                    if len(src) < end:
                        return None
                    out += src[i:i + size]
                    i = end

            data = None
            while data is None:
                if chunked:
                    data = dechunk(body)
                    if data is not None:
                        break
                elif content_length and len(body) >= content_length:
                    data = bytes(body[:content_length])
                    break

                chunk = sock.recv(65536)
                if not chunk:
                    if chunked:
                        raise ConnectionError(
                            f"Chunked incomplete: {len(body)}"
                        )
                    data = bytes(body)  # EOF-delimited
                    break
                body += chunk

            total = len(data)
            print(f"[RAW] Received {total} bytes")

            if total == 0:
                raise ConnectionError("Empty body")

            # Self-correct dims if header values are wrong
            if width * height * 2 != total:
                px_total = total // 2
                if px_total % width == 0:
                    height = px_total // width
                elif px_total % height == 0:
                    width = px_total // height
                print(
                    f"[RAW] Dim mismatch — corrected to "
                    f"{width}x{height}"
                )

            # Reshape — zero-copy, NO jpeg decode
            # RGB565: 2 bytes/pixel, 16-bit little-endian words
            px = np.frombuffer(
                data, dtype="<u2"
            ).reshape(height, width)

            # RGB565 -> BGR for YOLO (expects OpenCV BGR)
            frame_bgr = np.stack([
                ((px >> 11) & 0x1F) << 3,
                ((px >> 5) & 0x3F) << 2,
                (px & 0x1F) << 3,
            ], axis=-1).astype(np.uint8)

            return frame_bgr

        except Exception as e:
            print(f"[RAW] Error: {e}")
            return None
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def close(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


# ============================================================
# UDP NOTIFIER
# ============================================================

class UdpNotifier:

    def __init__(self, target_ip, target_port, cooldown=UDP_COOLDOWN):
        self.target_ip = target_ip
        self.target_port = target_port
        self.cooldown = cooldown
        self._last_sent = 0.0
        self._lock = threading.Lock()
        self._sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )

    def notify(self, message="POTHOLE"):
        now = time.perf_counter()
        with self._lock:
            if now - self._last_sent < self.cooldown:
                return
            try:
                self._sock.sendto(
                    message.encode(),
                    (self.target_ip, self.target_port)
                )
                self._last_sent = now
                print(
                    f"[UDP] Sent '{message}' "
                    f"to {self.target_ip}:{self.target_port}"
                )
            except OSError as exc:
                print(f"[UDP] Send error: {exc}")

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


# ============================================================
# ASYNC DETECTOR
# ============================================================

class AsyncDetector:

    def __init__(self, model_path, stream_url, raw_mode=False):

        print("Loading model...")

        # Start RAM tracking
        tracemalloc.start()
        self._process = psutil.Process()

        try:
            import torch
            torch.set_num_threads(max(1, os.cpu_count() or 4))
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except ImportError:
            self.device = "cpu"

        self.half = self.device == "cuda"
        if self.half:
            print("[SPEED] FP16 half-precision inference ENABLED")

        print(f"Using device: {self.device}")

        base = os.path.splitext(model_path)[0]
        ov_dir = base + "_openvino_model"
        if os.path.isdir(ov_dir):
            model_path = ov_dir
            print(f"[MODEL] Using OpenVINO: {ov_dir} (FASTEST on CPU)")
        elif not os.path.exists(model_path):
            model_path = base + ".pt"
            print(f"[MODEL] {model_path} not found — falling back to .pt")

        self.model = YOLO(model_path)

        # Snapshot model RAM
        current, peak = tracemalloc.get_traced_memory()
        self._model_ram_mb = peak / (1024 * 1024)
        print(f"[RAM] Model loaded — Peak: {self._model_ram_mb:.2f} MB")

        self.raw_mode = raw_mode
        self.stream_url = stream_url
        self._raw_reader = None
        self._sock = None
        self._buffer = b""
        self._decoded = b""

        if self.raw_mode:
            ip = stream_url.split("/")[2].split(":")[0]
            print("[MODE] RAW RGB565 — JPEG encode/decode ELIMINATED")
            self._raw_reader = RawFrameReader(ip)
        else:
            print("[MODE] JPEG stream — traditional decode path")
            self._connect()

        print("Stream Connected Successfully")

        # Timing
        self._read_ms = 0.0
        self._network_ms = 0.0
        self._decode_ms = 0.0
        self._pre_ms = 0.0
        self._inf_ms = 0.0
        self._post_ms = 0.0
        self._total_ms = 0.0

        # FPS
        self._inference_count = 0
        self._fps_start_time = time.perf_counter()
        self._inference_fps = 0.0
        self._frame_count = 0
        self._frame_fps_start_time = time.perf_counter()
        self._camera_fps = 0.0

        # Latest frame
        self._frame_id = 0
        self._latest_frame = None
        self._latest_frame_id = -1
        self._boxes = []
        self._lock = threading.Lock()
        self._stop = False

        # Threads
        if not self.raw_mode:
            self._reader_thread = threading.Thread(
                target=self._read_loop, daemon=True
            )
            self._reader_thread.start()

        self._infer_thread = threading.Thread(
            target=self._infer_loop, daemon=True
        )
        self._infer_thread.start()

        self._warmup()


    # ========================================================
    # WARMUP
    # ========================================================

    def _warmup(self):
        print("Warming up YOLO...")
        dummy = np.zeros(
            (INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8
        )
        for _ in range(3):
            self.model.predict(
                source=dummy,
                imgsz=INFER_SIZE,
                conf=CONF_THRESHOLD,
                device=self.device,
                max_det=MAX_DET,
                verbose=False
            )
        print("Warm-up complete.")


    # ========================================================
    # CONNECT (JPEG mode only)
    # ========================================================

    def _connect(self):
        ip = self.stream_url.split("/")[2]
        self._sock = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
        )
        self._sock.settimeout(5)
        self._sock.connect((ip, 80))
        self._sock.sendall(
            b"GET /stream HTTP/1.1\r\n"
            b"Host: " + ip.encode() + b"\r\n"
            b"Connection: keep-alive\r\n"
            b"\r\n"
        )
        self._buffer = b""
        self._decoded = b""
        self._no_data_count = 0

        data = b""
        while b"\r\n\r\n" not in data:
            data += self._sock.recv(1024)
        self._buffer = data[data.find(b"\r\n\r\n") + 4:]


    # ========================================================
    # CAMERA READER THREAD (JPEG mode only)
    # ========================================================

    def _read_loop(self):
        boundary = b"--123456789000000000000987654321"

        while not self._stop:
            try:
                if self._sock is None:
                    self._connect()

                ready = select.select(
                    [self._sock], [], [], 1.0
                )

                if ready[0]:
                    self._no_data_count = 0
                    chunk = self._sock.recv(8192)
                    if not chunk:
                        raise ConnectionError("Connection closed")
                    self._buffer += chunk
                else:
                    self._no_data_count += 1
                    if self._no_data_count > 5:
                        raise TimeoutError("No data for 5s")

            except Exception as exc:
                print(f"[STREAM] Reconnecting: {exc}")
                if self._sock:
                    try:
                        self._sock.close()
                    except Exception:
                        pass
                    self._sock = None
                time.sleep(0.5)
                try:
                    self._connect()
                except Exception:
                    self._sock = None
                continue

            # Decode HTTP chunked data
            try:
                work = self._buffer
                while True:
                    eol = work.find(b"\r\n")
                    if eol == -1:
                        break
                    try:
                        chunk_size = int(work[:eol], 16)
                    except ValueError:
                        break
                    if chunk_size == 0:
                        work = work[eol + 2:]
                        break
                    if len(work) < (eol + 2 + chunk_size + 2):
                        break
                    self._decoded += work[eol + 2:eol + 2 + chunk_size]
                    work = work[eol + 2 + chunk_size + 2:]
                self._buffer = work
            except Exception:
                self._decoded += self._buffer
                self._buffer = b""

            # Extract JPEG frame
            while True:
                start = self._decoded.find(
                    boundary + b"\r\nContent-Type: image/jpeg\r\n"
                )
                if start == -1:
                    break
                data_start = self._decoded.find(b"\r\n\r\n", start)
                if data_start == -1:
                    break
                data_start += 4
                end = self._decoded.find(boundary, data_start)
                if end == -1:
                    break
                jpeg_data = self._decoded[data_start:end]
                self._decoded = self._decoded[end:]

                if len(jpeg_data) > 100:
                    decode_start = time.perf_counter()

                    arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    decode_time = (
                        time.perf_counter() - decode_start
                    ) * 1000

                    if frame is not None:
                        with self._lock:
                            self._frame_id += 1
                            self._latest_frame = frame
                            self._latest_frame_id = self._frame_id
                            self._read_ms = decode_time
                            self._decode_ms = decode_time
                            self._frame_count += 1

                            elapsed = (
                                time.perf_counter()
                                - self._frame_fps_start_time
                            )
                            if elapsed >= 1.0:
                                self._camera_fps = (
                                    self._frame_count / elapsed
                                )
                                self._frame_count = 0
                                self._frame_fps_start_time = (
                                    time.perf_counter()
                                )


    # ========================================================
    # INFERENCE THREAD
    # ========================================================

    def _infer_loop(self):
        last_processed_id = -1

        while not self._stop:

            frame = None

            if self.raw_mode:
                # ============================================
                # RAW MODE — capture directly, zero decode
                # ============================================
                net_start = time.perf_counter()
                frame = self._raw_reader.capture()
                net_ms = (time.perf_counter() - net_start) * 1000

                if frame is None:
                    time.sleep(0.01)
                    continue

                with self._lock:
                    self._network_ms = net_ms
                    self._decode_ms = 0.0
                    self._read_ms = 0.0
                    self._frame_id += 1
                    self._latest_frame = frame
                    self._latest_frame_id = self._frame_id

            else:
                # ============================================
                # JPEG MODE — get from reader thread
                # ============================================
                with self._lock:
                    fid = self._latest_frame_id
                    if (
                        fid == last_processed_id
                        or self._latest_frame is None
                    ):
                        frame = None
                    else:
                        frame = self._latest_frame.copy()
                        last_processed_id = fid

                if frame is None:
                    time.sleep(0.003)
                    continue

            # ============================================
            # YOLO INFERENCE
            # ============================================

            total_start = time.perf_counter()

            results = self.model.predict(
                source=frame,
                imgsz=INFER_SIZE,
                conf=CONF_THRESHOLD,
                device=self.device,
                max_det=MAX_DET,
                verbose=False
            )

            total_ms = (
                time.perf_counter() - total_start
            ) * 1000

            # Ultralytics timings
            speed = results[0].speed
            pre_ms = speed.get("preprocess", 0)
            inf_ms = speed.get("inference", 0)
            post_ms = speed.get("postprocess", 0)

            # Extract pothole boxes
            boxes = []
            for r in results[0].boxes:
                cls = int(r.cls[0])
                if cls != 0:
                    continue
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                conf = float(r.conf[0])
                label = f"Pothole {conf:.2f}"
                boxes.append((x1, y1, x2, y2, label))

            # Update stats
            with self._lock:
                self._boxes = boxes
                self._pre_ms = pre_ms
                self._inf_ms = inf_ms
                self._post_ms = post_ms
                self._total_ms = total_ms

                self._inference_count += 1
                elapsed = (
                    time.perf_counter() - self._fps_start_time
                )
                if elapsed >= 1.0:
                    self._inference_fps = (
                        self._inference_count / elapsed
                    )
                    self._inference_count = 0
                    self._fps_start_time = time.perf_counter()


    # ========================================================
    # GETTERS
    # ========================================================

    def get_display_frame(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def get_boxes(self):
        with self._lock:
            return list(self._boxes)

    def get_stats(self):
        with self._lock:
            return (
                self._camera_fps,
                self._inference_fps,
                self._read_ms,
                self._network_ms,
                self._decode_ms,
                self._pre_ms,
                self._inf_ms,
                self._post_ms,
                self._total_ms,
            )

    def get_ram_stats(self):
        current, peak = tracemalloc.get_traced_memory()
        proc_mem = self._process.memory_info().rss
        return {
            "model_peak_mb": self._model_ram_mb,
            "trace_peak_mb": peak / (1024 * 1024),
            "process_rss_mb": proc_mem / (1024 * 1024),
        }


    # ========================================================
    # RELEASE
    # ========================================================

    def release(self):
        self._stop = True
        if not self.raw_mode:
            self._reader_thread.join(timeout=4)
        self._infer_thread.join(timeout=4)


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_boxes(frame, boxes, pothole_active):

    for (x1, y1, x2, y2, label) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            frame, (x1, y1 - th - 6), (x1 + tw + 4, y1),
            (0, 255, 0), -1
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    if pothole_active:
        cv2.putText(
            frame, "POTHOLE DETECTED",
            (frame.shape[1] // 2 - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Detect ESP32 camera format
    # --------------------------------------------------------

    if FORCE_RAW_MODE:
        raw_mode = True
        bench = None
    else:
        raw_mode = False
        bench = None

    # --------------------------------------------------------
    # Init detector
    # --------------------------------------------------------

    detector = AsyncDetector(
        MODEL_PATH,
        STREAM_URL,
        raw_mode=raw_mode
    )

    notifier = UdpNotifier(ESP8266_IP, UDP_PORT)

    # --------------------------------------------------------
    # History buffers
    # --------------------------------------------------------

    max_samples = 30

    camera_fps_hist = []
    inference_fps_hist = []
    network_hist = []
    read_hist = []
    decode_hist = []
    pre_hist = []
    inf_hist = []
    post_hist = []
    total_hist = []

    pothole_active = False

    try:
        while True:

            frame = detector.get_display_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            boxes = detector.get_boxes()

            (
                camera_fps,
                inference_fps,
                read_ms,
                network_ms,
                decode_ms,
                pre_ms,
                inf_ms,
                post_ms,
                total_ms,
            ) = detector.get_stats()

            pothole_active = len(boxes) > 0
            if pothole_active:
                notifier.notify()

            draw_boxes(frame, boxes, pothole_active)

            # Store history
            if camera_fps > 0:
                camera_fps_hist.append(camera_fps)
            if inference_fps > 0:
                inference_fps_hist.append(inference_fps)
            network_hist.append(network_ms)
            read_hist.append(read_ms)
            decode_hist.append(decode_ms)
            pre_hist.append(pre_ms)
            inf_hist.append(inf_ms)
            post_hist.append(post_ms)
            total_hist.append(total_ms)

            # Trim
            for h in [
                camera_fps_hist, inference_fps_hist,
                network_hist, read_hist, decode_hist,
                pre_hist, inf_hist, post_hist, total_hist
            ]:
                if len(h) > max_samples:
                    h.pop(0)

            # Averages
            def avg(lst):
                return sum(lst) / len(lst) if lst else 0

            avg_cam = avg(camera_fps_hist)
            avg_inf_fps = avg(inference_fps_hist)
            avg_net = avg(network_hist)
            avg_read = avg(read_hist)
            avg_dec = avg(decode_hist)
            avg_pre = avg(pre_hist)
            avg_inf = avg(inf_hist)
            avg_post = avg(post_hist)
            avg_total = avg(total_hist)

            cv2.imshow("ESP32-CAM Pothole Detection", frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    except KeyboardInterrupt:
        pass

    finally:
        detector.release()
        notifier.close()
        cv2.destroyAllWindows()

        # ====================================================
        # FINAL BOTTLENECK REPORT
        # ====================================================

        ram = detector.get_ram_stats()

        print()
        print("=" * 55)
        print("  FINAL PERFORMANCE & BOTTLENECK REPORT")
        print("=" * 55)
        print()

        print("  [CAMERA]")
        if bench:
            bench.print_stats()
        else:
            print("  (bench skipped — FORCE_RAW_MODE)")
        print()

        print("  [PYTHON MEMORY]")
        print(f"  Model RAM (peak)    : {ram['model_peak_mb']:.2f} MB")
        print(f"  Trace RAM (peak)    : {ram['trace_peak_mb']:.2f} MB")
        print(f"  Process RSS         : {ram['process_rss_mb']:.2f} MB")
        print()

        print("  [FPS]")
        if camera_fps_hist:
            print(f"  Camera FPS (avg)    : {avg(camera_fps_hist):.2f}")
        if inference_fps_hist:
            print(f"  Inference FPS (avg) : {avg(inference_fps_hist):.2f}")
        print()

        print("  [LATENCY BREAKDOWN]")
        if raw_mode:
            print(f"  Network Read        : {avg(network_hist):.2f} ms")
            print(f"  JPEG Decode         : 0.00 ms  (ELIMINATED)")
        else:
            if read_hist:
                print(f"  JPEG Decode (avg)   : {avg(read_hist):.2f} ms")
        if pre_hist:
            print(f"  Preprocess (avg)    : {avg(pre_hist):.2f} ms")
        if inf_hist:
            print(f"  Inference (avg)     : {avg(inf_hist):.2f} ms")
            print(f"  Inference (min)     : {min(inf_hist):.2f} ms")
            print(f"  Inference (max)     : {max(inf_hist):.2f} ms")
        if post_hist:
            print(f"  Postprocess (avg)   : {avg(post_hist):.2f} ms")
        if total_hist:
            print(f"  TOTAL (avg)         : {avg(total_hist):.2f} ms")
            print(f"  TOTAL (min)         : {min(total_hist):.2f} ms")
            print(f"  TOTAL (max)         : {max(total_hist):.2f} ms")
        print()

        print("  [BOTTLENECK]")
        stages = {}
        if network_hist:
            stages["Network Read"] = avg(network_hist)
        if decode_hist:
            stages["JPEG Decode"] = avg(decode_hist)
        if pre_hist:
            stages["Preprocess"] = avg(pre_hist)
        if inf_hist:
            stages["Inference"] = avg(inf_hist)
        if post_hist:
            stages["Postprocess"] = avg(post_hist)

        if stages:
            bottleneck = max(stages, key=stages.get)
            print(f"  Biggest bottleneck   : {bottleneck} ({stages[bottleneck]:.2f} ms)")

        if raw_mode and decode_hist:
            saved = sum(decode_hist) / len(decode_hist) if decode_hist else 0
            print(f"  JPEG decode saved   : ~{saved:.2f} ms per frame")

        print()
        print("=" * 55)


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
