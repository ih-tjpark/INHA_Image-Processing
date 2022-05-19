import cv2
import numpy as np

src = cv2.imread('./data/flower.jpg')
hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)
lowerb = (150, 100, 100)
upperb = (180, 255, 255)
dst = cv2.inRange(hsv, lowerb, upperb)


mode = cv2.RETR_CCOMP
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierarchy = cv2.findContours(dst, mode, method)


print('hierarchy', hierarchy.shape)
print('len(contours)=', len(contours))
print('contours[0].shape=', contours[0].shape)
print('contours=', contours)


for cnt in contours:
    cv2.drawContours(src, [cnt], 0, (255,0,0), 3)

cv2.imshow('src',  src)
cv2.waitKey()
cv2.destroyAllWindows()
