from ultralytics import YOLO
import cv2
import time

# Load your trained model
model = YOLO("best (1).pt")

# ESP32-CAM stream URL
url = "http://10.170.93.30/stream"

# Open stream
cap = cv2.VideoCapture(url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Target FPS
target_fps = 10
frame_time = 1.0 / target_fps

if not cap.isOpened():
    print("Cannot connect to ESP32-CAM")
    exit()

while True:

    start = time.time()

    ret, frame = cap.read()

    if not ret:
        print("Failed to receive frame")
        break

    # Resize frame for faster detection
    small = cv2.resize(frame, (320, 240))

    # Run YOLO
    results = model(small, conf=0.65, verbose=False)

    # Draw detections
    annotated = results[0].plot()

    # Enlarge for display
    annotated = cv2.resize(annotated, (640, 480))

    # Calculate FPS
    elapsed = time.time() - start
    fps = 1.0 / elapsed if elapsed > 0 else 0

    # Show FPS
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # Show video
    cv2.imshow("ESP32-CAM Pothole Detection", annotated)

    # Maintain target FPS
    if elapsed < frame_time:
        time.sleep(frame_time - elapsed)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()