import cv2
cap = cv2.VideoCapture(0)
while True:
    ret,frame = cap.read()
    if not ret:
        print("video not captured")
        break
    cv2.imshow("captured frame",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("quitting")
        break
cap.release()
cv2.destryALLWindow()
    