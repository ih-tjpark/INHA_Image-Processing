import cv2
import numpy as np

def click(event, x, y, flags, param):
    global refPt

    # 왼쪽 마우스가 클릭되면 (x, y) 좌표 기록을 시작
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(refPt) < 4:
            refPt.append((x, y))
            print(refPt)
        else:
            print('reset')
            refPt = []

def warp(img):
    global pts1, pts2, pts3, pts4, dst, perspect_mat
    perspect_mat =cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img, perspect_mat, img.shape[1::-1], cv2.INTER_CUBIC)

img = cv2.imread('./data/images/perspective2.jpg')
if img is None: raise Exception("영상파일 읽기 에러")

# 새 윈도우 창을 만들고 그 윈도우 창에 click_and_crop 함수를 세팅해 줍니다.
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

refPt = []

pts1 = np.float32(refPt)
pts2 = np.float32([(50,60), (340,60), (50,320), (340,320)])
clone = img.copy()
dst = img


'좌측 상단 -> 우측 상단 -> 좌측 하단 -> 우측 하단 순으로 클릭'
while True:

    key = cv2.waitKey(1) & 0xFF
    pts1 = np.float32(refPt)

    # 클릭시 이미지에 마크 출력
    if len(pts1) >=1:
        for i in range(len(pts1)):
            cv2.circle(img, tuple(pts1[i].astype(int)), 3, (0, 255, 0), -1)

    # 4개의 점 이상을 선택하면 reset
    else:
        img = clone.copy()
        dst = clone.copy()

    cv2.imshow("image", img)

    # 4개의 점이 선택되면 해당영역을 원근 투시 변환
    if len(pts1) == 4:
        warp(img)
        cv2.imshow('dst', dst)

    # 만약 q가 입력되면 작업을 끝냅니다.
    elif key == ord("q"):
        break


cv2.destroyAllWindows()
