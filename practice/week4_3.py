import cv2
import numpy as np
from matplotlib import pyplot as plt


im1 = cv2.imread('./data/im1.png')
im2 = cv2.imread('./data/im2.png')
im3 = cv2.imread('./data/im3.png')
im4 = cv2.imread('./data/im4.png')
im5 = cv2.imread('./data/im5.png')


def hist(img):
    H = cv2.calcHist(images=img, channels=[0], mask=None,
                  histSize=[256], ranges = [0,255])
    cv2.normalize(H,H,1,0,cv2.NORM_L1)

    return H

H1 = hist(im1)
H2 = hist(im2)
H3 = hist(im3)
H4 = hist(im4)
H5 = hist(im5)

d1 = cv2.compareHist(H1, H2, cv2.HISTCMP_CORREL)
d2 = cv2.compareHist(H1, H3, cv2.HISTCMP_CORREL)
d3 = cv2.compareHist(H1, H4, cv2.HISTCMP_CORREL)
d4 = cv2.compareHist(H1, H5, cv2.HISTCMP_CORREL)

print('d1(H1, H2, CORREL) =',       d1)
print('d2(H1, H3, CORREL) =',       d2)
print('d3(H1, H4, CORREL) =',       d3)
print('d4(H1, H5, CORREL) =',       d4)


if max(d1,d2,d3,d4) == d1:
    print("img1, img2 가장 유사")
elif max(d1,d2,d3,d4) == d2:
    print("img1, img3 가장 유사")
elif max(d1,d2,d3,d4) == d3:
    print("img1, img4 가장 유사")
elif max(d1,d2,d3,d4) == d4:
    print("img1, img5 가장 유사")
else: print(" ")







