import cv2
import time
import threading
import socket
import select
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best (1).pt"
STREAM_URL = "http://172.16.2.3/stream"
CONF_THRESHOLD = 0.65
INFER_SIZE = 320

ESP8266_IP = "172.16.10.80"
UDP_PORT = 4210
UDP_COOLDOWN = 5.0


class UdpNotifier:
    def __init__(self, target_ip, target_port, cooldown=UDP_COOLDOWN):
        self.target_ip = target_ip
        self.target_port = target_port
        self.cooldown = cooldown
        self._last_sent = 0.0
        self._lock = threading.Lock()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def notify(self, message="POTHOLE"):
        now = time.monotonic()
        with self._lock:
            if now - self._last_sent < self.cooldown:
                return
            try:
                self._sock.sendto(message.encode(), (self.target_ip, self.target_port))
                self._last_sent = now
                print(f"[UDP] Sent '{message}' to {self.target_ip}:{self.target_port}")
            except OSError as exc:
                print(f"[UDP] Send error: {exc}")

    def close(self):
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class AsyncDetector:
    def __init__(self, model_path, stream_url):
        print("Loading model...")
        self.model = YOLO(model_path)

        print(f"Connecting to {stream_url} ...")
        self.stream_url = stream_url
        self._sock = None
        self._buffer = b""
        self._connect()
        print("Stream Connected Successfully")

        self._raw_frame = None
        self._decoded = b""
        self._boxes = []
        self._lock = threading.Lock()
        self._stop = False

        self._reader_thread = threading.Thread(target=self._read_loop)
        self._reader_thread.start()

        self._infer_thread = threading.Thread(target=self._infer_loop)
        self._infer_thread.start()

        self._read_time = 0.0
        self._infer_time = 0.0

    def _connect(self):
        ip = self.stream_url.split("/")[2]
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5)
        self._sock.connect((ip, 80))
        self._sock.sendall(
            b"GET /stream HTTP/1.1\r\n"
            b"Host: " + ip.encode() + b"\r\n"
            b"Connection: keep-alive\r\n"
            b"\r\n"
        )
        self._buffer = b""
        self._no_data_count = 0
        data = b""
        while b"\r\n\r\n" not in data:
            data += self._sock.recv(1024)
        self._buffer = data[data.find(b"\r\n\r\n") + 4:]

    def _read_loop(self):
        boundary = b"--123456789000000000000987654321"
        while not self._stop:
            t0 = time.time()
            try:
                if self._sock is None:
                    self._connect()
                ready = select.select([self._sock], [], [], 1.0)
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
            except Exception:
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
                    if len(work) < eol + 2 + chunk_size + 2:
                        break
                    self._decoded += work[eol + 2:eol + 2 + chunk_size]
                    work = work[eol + 2 + chunk_size + 2:]
                self._buffer = work
            except Exception:
                self._decoded += self._buffer
                self._buffer = b""

            while True:
                start = self._decoded.find(boundary + b"\r\nContent-Type: image/jpeg\r\n")
                if start == -1:
                    break
                data_start = self._decoded.find(b"\r\n\r\n", start) + 4
                end = self._decoded.find(boundary, data_start)
                if end == -1:
                    break
                jpeg_data = self._decoded[data_start:end]
                self._decoded = self._decoded[end:]
                if len(jpeg_data) > 100:
                    arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        read_ms = (time.time() - t0) * 1000
                        with self._lock:
                            self._raw_frame = frame
                            self._read_time = read_ms

    def _infer_loop(self):
        skip = 0
        while not self._stop:
            with self._lock:
                if self._raw_frame is None:
                    frame = None
                else:
                    frame = self._raw_frame.copy()
            if frame is None:
                time.sleep(0.01)
                continue
            skip += 1
            if skip < 3:
                continue
            skip = 0

            t0 = time.time()
            results = self.model.predict(
                source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False
            )
            infer_ms = (time.time() - t0) * 1000
            boxes = []
            for r in results[0].boxes:
                x1, y1, x2, y2 = map(int, r.xyxy[0])
                conf = float(r.conf[0])
                cls = int(r.cls[0])
                if cls != 0:
                    continue
                label = f"Pothole {conf:.2f}"
                boxes.append((x1, y1, x2, y2, label))

            with self._lock:
                self._boxes = boxes
                self._infer_time = infer_ms

    def get_frame_and_boxes(self):
        with self._lock:
            if self._raw_frame is None:
                return None, [], 0, 0
            return self._raw_frame.copy(), list(self._boxes), self._read_time, self._infer_time

    def release(self):
        self._stop = True
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._reader_thread.join(timeout=4)
        self._infer_thread.join(timeout=4)


def draw_boxes(frame, boxes, pothole_active):
    for x1, y1, x2, y2, label in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if pothole_active:
        cv2.putText(frame, "POTHOLE DETECTED", (frame.shape[1] // 2 - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def main():
    detector = AsyncDetector(MODEL_PATH, STREAM_URL)
    notifier = UdpNotifier(ESP8266_IP, UDP_PORT)
    prev = time.time()
    fps_hist = []
    read_hist = []
    infer_hist = []
    pothole_active = False

    try:
        while True:
            frame, boxes, read_ms, infer_ms = detector.get_frame_and_boxes()
            if frame is None:
                time.sleep(0.01)
                continue

            pothole_active = len(boxes) > 0
            if pothole_active:
                notifier.notify()

            draw_boxes(frame, boxes, pothole_active)

            h, w = frame.shape[:2]
            dt = time.time() - prev
            fps = 1 / dt if dt > 0 else 0
            prev = time.time()

            fps_hist.append(fps)
            read_hist.append(read_ms)
            infer_hist.append(infer_ms)
            if len(fps_hist) > 30:
                fps_hist.pop(0)
                read_hist.pop(0)
                infer_hist.pop(0)

            info = f"FPS:{sum(fps_hist)/len(fps_hist):.1f} READ:{sum(read_hist)/len(read_hist):.0f}ms INF:{sum(infer_hist)/len(infer_hist):.0f}ms"
            cv2.rectangle(frame, (5, h - 35), (w - 5, h - 5), (0, 0, 0), -1)
            cv2.putText(frame, info, (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("ESP32-CAM Pothole Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        detector.release()
        notifier.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
