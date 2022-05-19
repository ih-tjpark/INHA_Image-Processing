import cv2

src = cv2.imread('./data/face.jpg')
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
lowerb = (0, 30, 85)
upperb = (20, 180, 255)
dst = cv2.inRange(hsv, lowerb, upperb)


cv2.imshow('src',  src)
cv2.imshow('dst',  dst)

cv2.waitKey()
cv2.destroyAllWindows()
'''
한장의 얼굴사진을 검출하려고 하면 그에 맞는 hsv의 상한값과 하한값을 지정해줘야 정확히 검출이 됩니다.
같은 코드로 다른 얼굴사진을 검출 시 상/하한가를 다시 셋팅해줘야 되서 시간적으로 비효율적이고 여러장을 동시에 검출할 수 없습니다.
그래서 얼굴영역의 색상을 감출하여 hsv를 자동으로 셋팅하거나 
이미지를 쉽게 인식할 수 있도록 흑백 이미지를 정규화 후 검출하는 방식을 고안할 필요가 있습니다.
자동으로 하는건 난이도가 높을것으로 예상되어 예전에 배웠던 히스토그램 이퀄라이제이션으로 정규화를 해서 범위안에 들 수 있도록 하는 방법이 
제가 구현할 수 있는 코드 수준에서는 현실적이라고 생각합니다.
'''