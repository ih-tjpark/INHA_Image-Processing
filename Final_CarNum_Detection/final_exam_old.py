import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import imutils
import cv2
img = cv2.imread('data/carNum/z7.jpg')
img = imutils.resize(img, width=500 )
img_t = img.copy()
imgH, imgW, imgC = img.shape

#gray 스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 가우시안 블러로 노이즈 제거
#blurr = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
blurr = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200)
cv2.imshow("Canny",edged)


# 경계선 찾기
contours, _ = cv2.findContours(
    edged,
    method=cv2.CHAIN_APPROX_SIMPLE,
    mode= cv2.RETR_LIST
)
cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
# 번호판 글자 위치 찾기
img_temp = img.copy()
cv2.drawContours(img_temp,cnts,-1,(0,255,0),3)

cv2.imshow('a',img_temp)


temp_np = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
contours_dic =[]

# 경계선 좌표를 이용해 사각형 그리기
for con in contours:
    x, y, w, h = cv2.boundingRect(con)
    cv2.rectangle(temp_np, pt1=(x,y), pt2=(x+w, y+h),
                  color=(255,255,255),thickness=2)

    contours_dic.append({
        'contour': con,
        'x' : x,
        'y' : y,
        'w' : w,
        'h' : h,
        'cx' : x + (w / 2),
        'cy' : y + (h / 2)
    })
cv2.imshow('all_contour',temp_np)



MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dic:
    #직사각형 넓이
    area = d['w'] * d['h']
    #가로 세로 비율
    ratio = d['w'] / d['h']


    if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

temp_result = np.zeros((imgH, imgW, imgC), dtype = np.uint8)

for d in possible_contours:

    cv2.rectangle(temp_result,
                  (d['x'], d['y']),
                  (d['x']+d['w'],
                   d['y']+d['h']),
                  (255, 255, 255),
                  2)
cv2.imshow('possible_contour',temp_result)

MAX_DIAG_MULTIPLYER = 5 # 대각선의 길이 5배
MAX_ANGLE_DIFF = 12.0 # 1과 2 컨투어의 각도 차이 최댓값
MAX_AREA_DIFF = 0.5 # 면적 차이
MAX_WIDTH_DIFF = 0.8 # 너비 차이
MAX_HEIGHT_DIFF = 0.2 # 높이 차이
MIN_N_MATCHED = 4 # 만족하는 그룹이 최소 3개 이상

# 찾기
def find_chars(contour_list):

    # 최종 결과 인덱스를 저장
    matched_result_idx = []

    # 외곽선 1과 2를 비교
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:

            # 같은 컨투어는 비교할 필요 없음
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            # 중앙점 값을 구해서 컨투어 끼리의 대각 길이 구하기
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            # 각도 비교
            # (dx=0이면 90도) 90도면 번호판이 아닐 확률이 높아 제외
            if dx == 0:
                angle_diff = 90
            else:
                # 각도 구하기
                angle_diff = np.degrees(np.arctan(dy / dx))
            # 넓이, 가로, 세로 비율 구하기
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            # 파라미터에 만족하는 인덱스 추가
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        matched_contours_idx.append(d1['idx'])

        # 후보군을 뽑아 지정한 최소 갯수(3) 보다 낮으면 제외
        if len(matched_contours_idx) <= MIN_N_MATCHED:
            continue

        # 최종 후보군  컨투어 저장
        matched_result_idx.append(matched_contours_idx)

        # 후보군이 아닌 컨투어끼리 비교하기 위해 저장
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        # 후보군에서 인덱스 값만 추출
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # 재귀 함수
        recursive_contour_list = find_chars(unmatched_contour)

        # 살아남은 후보군 저장
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break
    print(matched_result_idx)
    return matched_result_idx

# 외곽선중에 번호판 후보군 추출
#print(possible_contours)
result_idx = find_chars(possible_contours)

# 최종 후보군 저장
matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# 최종 후보군 사각형 표시
temp_result = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)
        cv2.rectangle(img, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)

cv2.imshow('final contour',temp_result)

PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []
#print(matched_result)


# 번호판 영역 각도 변경 (affine transform)
for i, matched_chars in enumerate(matched_result):
    # x 방향으로 순차 오름 정렬
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    # 첫번째와 끝의 사각형 중심을 이용해 번호판 중심 좌표 계산
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    # 첫과 마지막 사각형의 끝과 끝을 잇는 x값 * padding으로 가로 길이 계산
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING


    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    # 사각형들의 높이 평균 * padding으로 세로 길이 계산
    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    # y값 센터로 각도를 구하기 위한 삼각형의 높이 계산
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']

    # 첫과 마지막 사각형 센터로 삼각형 대각선의 길이 계산
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )

    # 세타값 각도 구하기
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus)) -0.2

    # rotationMatrix와 warpAffine함수로 이미지 회전
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    #img_rotated = cv2.warpAffine(thresh, M=rotation_matrix, dsize=(imgW, imgH))

    img_t_rotated = cv2.warpAffine(img_t, M=rotation_matrix, dsize=(imgW, imgH))
    cv2.imshow('rotate',img_t_rotated)

    img_cropped2 = cv2.getRectSubPix(
        img_t_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )
    img_cropped2_gray = cv2.cvtColor(img_cropped2,cv2.COLOR_BGR2GRAY)
    ret ,img_cropped2_th = cv2.threshold(img_cropped2_gray, 160,255,cv2.THRESH_BINARY)
    cv2.imshow('rotate2',img_cropped2_th)

    plate_imgs.append(img_cropped2_th)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })


print(plate_imgs[1])
longest_idx, longest_text = -1, 0
plate_chars = []

# 번호판 이미지 재확인 및 전처리 후 번호 인식
for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    # 한번 더 비교
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        ratio = w / h

        if area > MIN_AREA \
                and w > MIN_WIDTH and h > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h

    # 최종 이미지 추출
    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]


    #cv2.imshow('test5',img_result)
    # 최종 이미지 전처리
    #img_result = cv2.GaussianBlur(img_result, ksize=(5, 5), sigmaX=0)
    #_, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))


    mask = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]]).astype('uint8')


    #dst = cv2.morphologyEx(img_result,  cv2.MORPH_CLOSE,mask, iterations=2)
    #dst = cv2.morphologyEx(img_result, cv2.MORPH_DILATE,mask, iterations=4)



    pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
    # psm 7 => 이미지에 문자가 한줄로 되어 있음 , 0번 legacy 사용(오래된 버전)
    chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
    print(chars)
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit() or (c =='_') or(c=='-'):
            if c.isdigit():
                has_digit = True
            result_chars += c



    plate_chars.append(result_chars)

    if has_digit and len(result_chars) > longest_text:
        longest_idx = i

    #cv2.imshow('result',img_result)



info = plate_infos[longest_idx]
chars = plate_chars[longest_idx]

print(chars)

cv2.rectangle(img_t_rotated, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(0,0,255), thickness=2)

cv2.imshow('final_result',img_t_rotated)

cv2.waitKey()
#print(img)

