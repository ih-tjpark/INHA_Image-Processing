# 0603.py
import cv2
import numpy as np
src = cv2.imread('./data/dragon.jpg', cv2.IMREAD_GRAYSCALE)

dst1 = cv2.GaussianBlur(src, ksize=(5,5), sigmaX=0)
cv2.imshow('dst1',  dst1)

gx = cv2.Sobel(dst1, cv2.CV_32F, 1, 0, ksize = 3)
gy = cv2.Sobel(dst1, cv2.CV_32F, 0, 1, ksize = 3)


mag = cv2.magnitude(gx, gy)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mag)
print('mag:', minVal, maxVal, minLoc, maxLoc)
dstM = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('src',  src)
cv2.imshow('dstM',  dstM)

cv2.imwrite('dragon.jpg',dstM)

cv2.waitKey()
cv2.destroyAllWindows()
