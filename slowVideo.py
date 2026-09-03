import cv2

# Input video
cap = cv2.VideoCapture("mixkit-potholes-in-a-rural-road-25208-hd-ready.mp4")

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
out = cv2.VideoWriter(
    "slow_output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Write each frame twice (2x slower)
    out.write(frame)
    out.write(frame)

cap.release()
out.release()

print("Slow-motion video saved as slow_output.mp4")