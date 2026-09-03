import cv2
image = cv2.imread("dearImage.jpg")
h,w,c = image.shape
# resized = cv2.resize(image,(200,200))
# cv2.imshow("hello",resized)
# cv2.imwrite("dearImage.jpg",resized)

# print(f"{h}\n{w}\n{c}")
# cv2.waitKey(0)
# cv2.destroyAllWindow()

M = cv2.getRotationMatrix2D((w//2,h//2),45,1.0)
rotatedImage = cv2.warpAffine(image,M,(w,h))
cv2.imshow("helllo",rotatedImage)

cv2.waitKey(0)
cv2.destroyAllWindow()