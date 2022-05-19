import cv2,pafy
import numpy as np
import math

#사진및 비디오 불러오기
src = cv2.imread('./data/sample.jpg.png')
video1 = cv2.VideoCapture('./data/vclip1.mp4')
video2 = cv2.VideoCapture('./data/vclip2.mp4')
video3 = cv2.VideoCapture('./data/vclip3.mp4')



# 히스토그램 생성 함수
def hist(img):
    H = cv2.calcHist(images=img, channels=[0], mask=None,
                     histSize=[256], ranges= [0,255])
    cv2.normalize(H,H,1,0,cv2.NORM_L1)
    return H

while(True):
    # 프레임 객체 획득
    retval1, frame1 = video1.read()
    retval2, frame2 = video2.read()
    retval3, frame3 = video3.read()

    #정상적으로 획득했는지 체크
    if not retval1 and retval2 and retval3:
        break

    #각 영상의 1프레임 획득
    frame_set1 = video1.get(cv2.CAP_PROP_POS_FRAMES)
    frame_set2 = video2.get(cv2.CAP_PROP_POS_FRAMES)
    frame_set3 = video3.get(cv2.CAP_PROP_POS_FRAMES)
    if (frame_set1 == 1 ):
        video1_frame1 = frame1
    if (frame_set2 == 1 ):
        video2_frame1 = frame2
    if (frame_set3 == 1 ):
        video3_frame1 = frame3

    #반복문 나가기
    if frame_set1 >=1: break

# 1프레임 히스토그램 변환
src_hist = hist(src)
video1_hist = hist(video1_frame1)
video2_hist = hist(video2_frame1)
video3_hist = hist(video3_frame1)

#샘플 사진과 각 영상 1프레임 히스토그램 비교
comp1 = cv2.compareHist(src_hist,video1_hist,cv2.HISTCMP_CORREL)
comp2 = cv2.compareHist(src_hist,video2_hist,cv2.HISTCMP_CORREL)
comp3 = cv2.compareHist(src_hist,video3_hist,cv2.HISTCMP_CORREL)
print('Video 1=', comp1)
print('Video 2=', comp2)
print('Video 3=', comp3)
print("")

#유사한 비디오 선택 및 객체 저장
comp_max = max(comp1,comp2,comp3)
if comp_max ==comp1:
    print("video1이 가장 유사")
    video = video1
    video_f = video1_frame1
elif comp_max ==comp2:
    print("video2이 가장 유사")
    choice=2
    video = video2
    video_f = video2_frame1
elif comp_max ==comp3:
    print("video3이 가장 유사")
    choice=3
    video = video3
    video_f = video3_frame1
else:
    print('유사한 비디오 없음')
    video = None

#선택된 영상을 그레이스케일로 변환 후 영상에 맞게 old_frame 초기화
gray_video_f = cv2.cvtColor(video_f,cv2.COLOR_BGR2GRAY)
h,w =gray_video_f.shape
old_frame = np.zeros(shape=(h,w),dtype=np.uint8)

#기울기 구하기 위한 변수 값 지정 (거리, 각도)
rho, theta = 1,  np.pi/180

#라인 그리는 함수
def draw_houghLines(src, lines, nline):
    # 컬러 영상 변환
    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    min_length = min(len(lines), nline)
    for i in range(min_length):
        # 수직거리 , 각도 - 3차원 행렬
        rho, radian = lines[i, 0, 0:2]
        a, b = math.cos(radian), math.sin(radian)
        # 검출 직선상의 한 좌표 계산
        pt = (a * rho, b * rho)
        # 직선상의 이동 위치
        delta = (-1000 * b, 1000 * a)
        pt1 = np.add(pt, delta).astype('int')
        pt2 = np.subtract(pt, delta).astype('int')
        cv2.line(dst, tuple(pt1), tuple(pt2), (0, 255, 0), 2, cv2.LINE_AA)
    return dst

while(True):
    retval,frame = video.read()

    if not retval:
        break

    # 현재 프레임 회색영상으로 변경
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # 1프레임 이상 진행된 후 비교작업 하기 위해 프레임 획득
    frame_set = video.get(cv2.CAP_PROP_POS_FRAMES)

    # 1프레임 이상인지 확인
    if frame_set >=1 :

        # 이전 프레임과 현재프레임 비교해 움직이는 부분 검출
        diff_frame = cv2.subtract(old_frame, gray)

        # 캐니에지 밝기 50이상 검출출
        canny = cv2.Canny(diff_frame,50,50)

        # houghLines을 통해 라인 구하기
        lines = cv2.HoughLines(canny, rho, theta, 80)

        # 라인을 찾을 경우에만 라인 그리기
        if type(lines) != type(None):
            dst = draw_houghLines(canny,lines,7)
        else: dst = canny # 라인 못찾으면 캐니에지만 된걸로 대체

        cv2.imshow('detected lines', dst)
        #cv2.imshow('diff_frame', canny)

    # 현재 frame은 다음 프레임과 비교하기 위해 old_frame으로 저장
    old_frame = gray


    # esc 누르면 영상 종료
    key = cv2.waitKey(25)
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()







