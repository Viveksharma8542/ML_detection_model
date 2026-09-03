import cv2
image = cv2.imread("dearImage.jpg")
cv2.line(image,(10,10),(100,100),(255,0,0),2)
cv2.rectangle(image,(10,10),(100,100),(255,0,0),2)
cv2.circle(image,(45,45),45,(0,255,0),2)
cv2.putText(image,"dear123",(90,90),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
cv2.imshow("hyy",image)
cv2.waitKey(0)
cv2.destroyAllWindow()