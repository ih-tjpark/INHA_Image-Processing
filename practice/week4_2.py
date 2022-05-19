import cv2


src = cv2.imread('./data/h2.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)
print(v)
dst = cv2.equalizeHist(v)

print(dst)

m_hsv = cv2.merge([h,s,dst])

bgr = cv2.cvtColor(m_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('v',v)
cv2.imshow('bgr',bgr)
cv2.imwrite('equlize.jpg',bgr)
cv2.waitKey()
cv2.destroyAllWindows()
