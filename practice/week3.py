import cv2
import numpy as np
src = cv2.imread('./data/dragon.jpg')

h, w, c = src.shape

print(h, w, c)

ycrcb = np.zeros(shape = (h,w,c) , dtype=np.uint8)

cv2.imshow('origin image',src)

for i in range(h-1):
    for j in range(w-1):
        y = (0.299 * src[i,j,2])\
            + (0.587 * src[i,j,1])\
            + (0.114 * src[i,j,0])
        cb = ((src[i,j,0] - y) * 0.564) + 128
        cr = ((src[i,j,2] - y) * 0.713) + 128

        ycrcb[i,j,0] = y
        ycrcb[i,j,1] = cr
        ycrcb[i,j,2] = cb


cv2.imshow('ycrcb',ycrcb)

dst = cv2.split(ycrcb)
cv2.imshow('ycrcb_y', dst[0])

cv2.waitKey()
cv2.destroyAllWindows()

