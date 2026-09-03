import time
import threading
import tracemalloc
import psutil
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "best (1).pt"
INFER_SIZE = 320
CONF_THRESHOLD = 0.65
WARMUP = 10
ITERATIONS = 100

model = YOLO(MODEL_PATH)

def count_parameters(model):
    total = 0
    for p in model.model.parameters():
        total += p.numel()
    return total

params = count_parameters(model)
print(f"Total parameters: {params:,}")
print(f"Model file size: {__import__('os').path.getsize(MODEL_PATH) / 1024 / 1024:.2f} MB")

frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

for _ in range(WARMUP):
    model.predict(source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)

tracemalloc.start()
model.predict(source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)
tracemalloc.stop()
tracemalloc.start()
for _ in range(ITERATIONS):
    model.predict(source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Python-side peak memory (tracemalloc): {peak / 1024 / 1024:.2f} MB")
print(f"Process RSS (psutil): {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")

lats = []
for _ in range(ITERATIONS):
    t0 = time.perf_counter()
    model.predict(source=frame, imgsz=INFER_SIZE, conf=CONF_THRESHOLD, verbose=False)
    lats.append((time.perf_counter() - t0) * 1000)

lats.sort()
min_ms = lats[0]
avg_ms = sum(lats) / len(lats)
p50 = lats[len(lats) // 2]
p95 = lats[int(len(lats) * 0.95)]
max_ms = lats[-1]

print(f"Inference per frame -> min: {min_ms:.1f} ms | avg: {avg_ms:.1f} ms | p50: {p50:.1f} ms | p95: {p95:.1f} ms | max: {max_ms:.1f} ms")
print(f"FPS (1/avg): {1000 / avg_ms:.1f}")
print(f"Throughput: {1000 / avg_ms:.1f} frames/sec")
print(f"Giga operations estimate (FLOPs, 320x320): {model.model.cuda() if False else 'see model_info'}")
try:
    info = model.info()
    print(f"model.info(): {info}")
except Exception as e:
    print(f"model.info failed: {e}")
