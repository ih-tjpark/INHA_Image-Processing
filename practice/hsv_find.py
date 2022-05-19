import cv2
import numpy as np

#trackbar callback fucntion to update HSV value
def callback(x):
    global H_low,H_high,S_low,S_high,V_low,V_high
    #assign trackbar position value to H,S,V High and low variable
    H_low = cv2.getTrackbarPos('low H','controls')
    H_high = cv2.getTrackbarPos('high H','controls')
    S_low = cv2.getTrackbarPos('low S','controls')
    S_high = cv2.getTrackbarPos('high S','controls')
    V_low = cv2.getTrackbarPos('low V','controls')
    V_high = cv2.getTrackbarPos('high V','controls')


#create a seperate window named 'controls' for trackbar
cv2.namedWindow('controls',2)
cv2.resizeWindow("controls", 550,10);


#global variable
H_low = 24
H_high = 89
S_low= 14
S_high = 115
V_low= 109
V_high = 255

#create trackbars for high,low H,S,V
cv2.createTrackbar('low H','controls',H_low,179,callback)
cv2.createTrackbar('high H','controls',H_high,179,callback)

cv2.createTrackbar('low S','controls',S_low,255,callback)
cv2.createTrackbar('high S','controls',S_high,255,callback)

cv2.createTrackbar('low V','controls',V_low,255,callback)
cv2.createTrackbar('high V','controls',V_high,255,callback)

#read source image
img=cv2.imread("./data/lena.jpg")
#convert sourece image to HSC color mode
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

while(1):

    #
    hsv_low = np.array([H_low, S_low, V_low], np.uint8)
    hsv_high = np.array([H_high, S_high, V_high], np.uint8)

    #making mask for hsv range
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    print (mask)
    #masking HSV value selected color becomes black
    res = cv2.bitwise_and(img, img, mask=mask)


    #show image
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)

    #waitfor the user to press escape and break the while loop
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

#destroys all window
cv2.destroyAllWindows()