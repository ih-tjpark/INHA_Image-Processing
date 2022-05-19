import numpy as np
import cv2,pafy
import math


url = 'https://www.youtube.com/watch?v=Bfxun1tuM20'
video = pafy.new(url)
best = video.getbest(preftype='mp4')

cap = cv2.VideoCapture(best.url)
mode = cv2.RETR_EXTERNAL
method = cv2.CHAIN_APPROX_SIMPLE

rho, theta = 1, np.pi/180 # 허프변환 거리,각도 간격
def draw_houghLines(src,lines, nline):
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    min_length = min(len(lines),nline)
    for i in range(min_length):
        rho2, radian = lines[i, 0, 0:2]
        a, b = math.cos(radian), math.sin(radian)
        pt = (a * rho2, b * rho2)
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt,delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2),(0,255,0),2,cv2.LINE_AA)
    return dst

def detect_maxObject(img):
    results = cv2.findContours(img,mode,method)
    contours = results[0]

    area = [cv2.contourArea(c) for c in contours]
    idx = np.argsort(area)
    max_rect = contours[idx[-1]]
    rect = cv2.boundingRect(max_rect)
    rect = np.add(rect, (-10, -10, 20, 20))
    return rect

while(True):
    retval, frame = cap.read()

    if not retval:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _,th_gray = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.erode(th_gray,kernel, iterations=1)

    x,y,w,h = detect_maxObject(np.copy(th_gray))
    roi = th_gray[y:y+h, x:x+w]

    canny = cv2.Canny(roi,40, 100)

    lines = cv2.HoughLines(canny,rho,theta,50) #기울기 계산
    print(lines,type(lines))


    if type(lines) != type(None):
        cv2.rectangle(morph, (x,y,w,h), 100, 2) #큰 객체 사각형 표시
        canny_line = draw_houghLines(canny, lines, 1)

        angle = (np.pi - lines[0, 0, 1] * 180 / np.pi)
        h, w  = frame.shape[:2]
        center = (w//2, h//2)
        rot_map = cv2.getRotationMatrix2D(center, -angle, 1)
        dst = cv2.warpAffine(frame, rot_map, (w,h), cv2.INTER_LINEAR)
        cv2.imshow('line', canny_line)
        cv2.imshow('dst',dst)


    cv2.imshow('frame',frame)



    #cv2.imshow('canny',canny)
    key = cv2.waitKey(25)
    if key == 27:
        break
