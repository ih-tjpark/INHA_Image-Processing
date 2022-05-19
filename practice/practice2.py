import cv2
import numpy as np

src = cv2.imread('./data/lena.jpg')

hsv = cv2.cvtColor(src,cv2.COLOR_BGR2HSV)

hsv_low = np.array([0, 0, 191], np.uint8)
hsv_high = np.array([69, 154, 255], np.uint8)

gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

mask = cv2.inRange(hsv,hsv_low,hsv_high)

mode = cv2.RETR_LIST #모든 윤곽선 검색, 계층 x
method = cv2.CHAIN_APPROX_SIMPLE
contours, hierachy = cv2.findContours(mask,mode,method)

print(contours)

for cnt in contours:
    cv2.drawContours(src, [cnt], 0, (255,0,0),3)

cv2.imshow('src',src)
cv2.waitKey()
cv2.destroyAllWindows()