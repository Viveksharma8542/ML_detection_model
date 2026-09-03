import cv2
Camera = cv2.VideoCapture(0)
width = int(Camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(Camera.get(cv2.CAP_PROP_FRAME_WIDTH))
codec = cv2.VideoWriter_fourcc(*'XVID')
recorder = cv2.VideoWriter("myvideo.mp4",codec,20,(width,height))
while True:
    success,frame = Camera.read()
    if not success:
        break
    recorder.write(frame)
    cv2.imshow("hello",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
Camera.release()
recorder.release()
cv2.destroyAllWindow()



