import cv2
import numpy as np

def erode_dliate(img, flag, mask=None):
    dst = np.zeros(img.shape, np.uint8)
    if mask is None: mask = np.ones((3,3), np.uint8)
    ycenter, xcenter = np.divmod(mask.shape[:2], 2)[0]

    mcnt = cv2.countNonZero(mask)

    for i in range(ycenter, img.shape[0] - ycenter):
        for j in range(xcenter, img.shape[1] - xcenter):
            y1, y2 = i - ycenter, i + ycenter + 1
            x1, x2 = j - xcenter, j + xcenter + 1
            roi = img[y1:y2, x1:x2]
            temp = cv2.bitwise_and(roi, mask)
            cnt = cv2.countNonZero(temp)
            if flag == 0: #침식 연산
                dst[i,j] = 255 if (cnt == mcnt) else 0
            if flag == 1: #침식 연산
                dst[i,j] = 0 if (cnt == 0) else 255
    return dst


src = cv2.imread('./data/car_num.png')


hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

#초록색 영역제외한 배경제거
lowerb1 = (24, 0, 122)
upperb1 = (85, 156, 255)
mask = cv2.inRange(hsv, lowerb1, upperb1)

res = cv2.bitwise_and(hsv, hsv, mask=mask)

#흰색 글자 추출
lowerb2 = (0, 0, 172)
upperb2 = (89, 28, 246)
dst = cv2.inRange(res, lowerb2, upperb2)

cv2.imshow('dst',dst)

mask = np.array([[0,1,0],
                [1,1,1],
                [0,1,0]]).astype('uint8')

th_img = cv2.threshold(dst, 128, 255, cv2.THRESH_BINARY)[1]

#erode_dliate(이미지, (0=erode / 1 = dliate), mask)
dst_m1 = erode_dliate(th_img, 1,mask)
dst_m2 = erode_dliate(dst_m1, 1,mask)
dst_m3 = erode_dliate(dst_m2, 0,mask)
dst_m4 = erode_dliate(dst_m3, 0,mask)
dst_m5 = erode_dliate(dst_m4, 1,mask)

#dst2 = cv2.morphologyEx(th_img,cv2.MORPH_CLOSE,mask,iterations=4)


cv2.imshow('aaa',dst_m5)
#cv2.imshow('bbb',dst2)

cv2.waitKey()
cv2.destroyAllWindows()
