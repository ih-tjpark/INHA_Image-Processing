import cv2
import numpy as np

def nonmax_suppression(sobel, direct): # 최대치 억제
    rows, cols = sobel.shape[:2]
    dst = np.zeros((rows, cols), np.float32)
    for i in range(1, rows - 1):
        for j in range(1, cols -1):
            values = sobel[i-1:i+2, j-1:j+2].flatten() #9개 화소 가져옴
            first = [3, 0, 1, 2] #처음 화소 좌표 4개
            id = first[direct[i, j]] #방향에 따른 첫 이웃화소 위치
            v1, v2 = values[id], values[8-id] #이웃화소 가져옴

            dst[i, j] = sobel[i, j] if (v1 < sobel[i,j] > v2) else 0
            # 중심화소가 이웃화소보다 작으면 억제 (최대치 억제)
    return dst

def _double_thresholding(g_suppressed, low, high):
    g_thresholded = np.zeros(g_suppressed.shape)
    for i in range(0, g_suppressed.shape[0]):
        for j in range(0, g_suppressed.shape[1]):
            if g_suppressed[i,j] < low:	# 하한가 보다 높은엣지 검출
                g_thresholded[i,j] = 0
            elif g_suppressed[i,j] >= low and g_suppressed[i,j] < high: 	# 약한엣지
                g_thresholded[i,j] = 128
            else:
                g_thresholded[i,j] = 255    # 강한엣지
    return g_thresholded

def _hysteresis(g_thresholded):
    g_strong = np.zeros(g_thresholded.shape)
    for i in range(0, g_thresholded.shape[0]):
        for j in range(0, g_thresholded.shape[1]):
            val = g_thresholded[i,j]
            if val == 128:			# 약한엣지인지 체크 후 이웃엣지가 강한엣지인지 체크
                if g_thresholded[i-1,j] == 255 or g_thresholded[i+1,j] == 255 or g_thresholded[i-1,j-1] == 255 or g_thresholded[i+1,j-1] == 255 or g_thresholded[i-1,j+1] == 255 or g_thresholded[i+1,j+1] == 255 or g_thresholded[i,j-1] == 255 or g_thresholded[i,j+1] == 255:
                    g_strong[i,j] = 255		# 약한엣지를 강한엣지로 변경
            elif val == 255:
                g_strong[i,j] = 255		# 강한엣지는 그대로 유지
    return g_strong


image = cv2.imread('./data/dragon.jpg', cv2.IMREAD_GRAYSCALE)


#Blur 처리
gaus_img = cv2.GaussianBlur(image,(5,5), 0.3)

# x,y 방향 마스크
gx = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 1, 0, ksize = 3)
gy = cv2.Sobel(np.float32(gaus_img), cv2.CV_32F, 0, 1, ksize = 3)

#두 행령 벡터 크기
sobel = cv2.magnitude(gx, gy)

directs = cv2.phase(gx, gy) / (np.pi/4) #에지 기울기 계산 및 근사
directs = directs.astype(int) % 4 # 8방향 -> 4방향 근사

max_sobel = nonmax_suppression(sobel, directs) #비최대치 억제

threshold= _double_thresholding(max_sobel, 100, 150) # 임계값 측정
hys = _hysteresis(threshold) # 높은 임계값 설정
cv2.imshow('hys', hys)


cv2.imshow('image',  image)


cv2.waitKey()
cv2.destroyAllWindows()
