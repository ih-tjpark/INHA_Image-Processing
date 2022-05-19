import cv2, pafy
import numpy as np
from matplotlib import pyplot as plt

url ='https://www.youtube.com/watch?v=FW2_CLNP9Oo'
video = pafy.new(url)
best = video.getbest(preftype='mp4')

cap = cv2.VideoCapture(best.url)

mode = cv2.RETR_CCOMP
method = cv2.CHAIN_APPROX_SIMPLE




while(True):
    retval, frame = cap.read()

    if not retval:
        break


    #색상 검출
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    low = (24,0,122)
    high = (85,156,255)
    dst = cv2.inRange(hsv,low,high)

    #검출된 색상으로 영상표시
    res = cv2.bitwise_and(frame,frame,mask=dst)
    cv2.imshow('res',res)

    #엣지 파란색으로 표시
    contours, hierarchy = cv2.findContours(dst, mode, method)
    #for cnt in contours:
        #cv2.drawContours(frame, [cnt], 0,(255,0,0),3)


    cv2.imshow('frame',frame)

    cv2.imshow('dst',dst)

    #엣지 검출
    #gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(dst,100,200)
    cv2.imshow('edges',edge)

    key = cv2.waitKey(25)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()