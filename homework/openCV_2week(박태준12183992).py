
import cv2
import numpy as np

img = cv2.imread('./data/lena.jpg') # cv2.IMREAD_COLOR
b_cal = 40
g_cal = 40
r_cal = -60

# Blue 계산
for y in range(100, 400):
    for x in range(200, 300):
        if((img[y, x, 0] + b_cal) > 255):
            img[y, x, 0] = 255
        elif ((img[y, x, 0] + b_cal) < 0):
            img[y, x, 0] = 0
        else: img[y, x, 0] += b_cal

# Green 계산
for y in range(100, 400):
    for x in range(300, 400):
        if((img[y, x, 1] + g_cal) > 255):
            img[y, x, 1] = 255
        elif ((img[y, x, 1] + g_cal) < 0):
            img[y, x, 1] = 0
        else: img[y, x, 1] += g_cal

# red 계산
for y in range(100, 400):
    for x in range(400, 500):
        if((img[y, x, 2] + r_cal) > 255):
            img[y, x, 2] = 255
        elif ((img[y, x, 2] + r_cal) < 0):
            img[y, x, 2] = 0
        else: img[y, x, 2] += r_cal

cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()


temp_img = np.zeros(shape=(255,255,255),dtype=np.uint8)

for x in range(0,512):
    for y in range(0,512):
        temp_img = img[x, y, 2]
        img[x, y, 2] = img[x, y, 1]
        img[x, y, 1] = temp_img

cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
