import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import imutils
from matplotlib import font_manager, rc
import os

font_path = "C:/font/HYKANB.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

# 이미지 전처리 (스케일 변환, 가우시안 블러, 캐니에지, 경계선 찾기)
def preprocessing(src_img):
    #gray 스케일 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러로 노이즈 제거
    blurr = cv2.GaussianBlur(gray, ksize=(3, 3), sigmaX=0)
    #blurr = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
    # 캐니에지로 외각선 검출
    edged = cv2.Canny(blurr, 30, 200)


    return edged

# 경계선 후보군 찾기
def possible_cont(edged_img):

    # 경계선 찾기
    contours, _ = cv2.findContours(
        edged_img,
        method=cv2.CHAIN_APPROX_SIMPLE,
        mode= cv2.RETR_LIST
    )
    temp_np = np.zeros((imgH, imgW, imgC), dtype=np.uint8)

    # 면적이 큰 순서대로 정렬 후 30개 추출
    cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:30]
    # 윤곽 그려주기
    cv2.drawContours(temp_np,cnts,-1,(0,255,0),3)


    temp_np = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
    contours_dic =[]

    # 경계선 좌표를 이용해 사각형 그리기
    for con in cnts:
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

    possible_contours = []
    cnt = 0
    # 사각형에서 번호판 후보군 찾기
    for d in contours_dic:
        #직사각형 넓이
        area = d['w'] * d['h']
        #가로 세로 비율
        ratio = d['w'] / d['h']

        # 넓이와 가로,세로 비율이 번호판 조건에 맞는지
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
    return possible_contours, temp_np, temp_result

# 디테일 하게 찾기
def find_chars(contour_list):
    MAX_DIAG_MULTIPLYER = 5     # 대각선의 길이 5배
    MAX_ANGLE_DIFF = 12.0       # 1과 2 컨투어의 각도 차이 최댓값
    MAX_AREA_DIFF = 0.8         # 면적 차이
    MAX_WIDTH_DIFF = 0.8        # 너비 차이
    MAX_HEIGHT_DIFF = 0.2       # 높이 차이
    MIN_N_MATCHED = 3           # 만족하는 그룹이 최소 3개 이상
    matched_result_idx = []     # 최종 결과 인덱스를 저장
    rec_in_cnt=[]
    # 외곽선 1과 2를 비교
    for d1 in contour_list:
        matched_contours_idx = []
        for count,d2 in enumerate(contour_list):

            # 같은 컨투어는 비교할 필요 없음
            if d1['idx'] == d2['idx']:
                continue

            if ((d2['x']>d1['x']) and (d2['x']<(d1['x']+d1['w'])))and\
                ((d2['y']>d1['y']) and (d2['y']<(d1['y']+d1['h']))):
                rec_in_cnt.append(d2['idx'])
                continue

            # 중앙점 값을 구해서 컨투어 끼리의 대각 길이 구하기
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))


            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
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

        # # 후보군이 아닌 컨투어끼리 비교하기 위해 저장
        # unmatched_contour_idx = []
        # for d4 in contour_list:
        #     if d4['idx'] not in matched_contours_idx:
        #         unmatched_contour_idx.append(d4['idx'])
        #
        # # 후보군에서 인덱스 값만 추출
        # unmatched_contour = np.take(possible_dic, unmatched_contour_idx)
        #
        # # 재귀 함수
        # recursive_contour_list = find_chars(unmatched_contour)

        # 살아남은 후보군 저장
        # for idx in recursive_contour_list:
        #     matched_result_idx.append(idx)

        break


    return matched_result_idx


# 각도변경 후 번호판 영역 추출
def angle(matched,th_row):
    PLATE_WIDTH_PADDING = 1.4 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 3
    MAX_PLATE_RATIO = 10

    plate_imgs = []
    plate_infos = []



    # 번호판 영역 각도 변경 (affine transform)
    for i, matched_chars in enumerate(matched):
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
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

        # rotationMatrix와 warpAffine함수로 이미지 회전
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        #img_rotated = cv2.warpAffine(thresh, M=rotation_matrix, dsize=(imgW, imgH))
        img_t = img.copy()
        img_rotated = cv2.warpAffine(img_t, M=rotation_matrix, dsize=(imgW, imgH))


        img_cropped = cv2.getRectSubPix(
            img_rotated,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )
        img_cropped_gray = cv2.cvtColor(img_cropped,cv2.COLOR_BGR2GRAY)
        img_blurr = cv2.GaussianBlur(img_cropped_gray,(3,3),0)

        ret ,img_cropped_th = cv2.threshold(img_blurr, th_row,255,cv2.THRESH_BINARY)



        plate_imgs.append(img_cropped_th)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
    return plate_imgs, plate_infos , img_rotated, img_cropped_th

# 문자 인식
def pytess(plate_imgs,plate_info):
    longest_idx, longest_text = -1, 0
    plate_chars = []

    # 번호판 이미지 재확인 및 전처리 후 번호 인식
    for i, plate_img in enumerate(plate_imgs):
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
        #_, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

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

        #img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=(0,0,0))

        pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
        # psm 7 => 이미지에 문자가 한줄로 되어 있음 , 0번 legacy 사용(오래된 버전)
        chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')
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



    info = plate_info[longest_idx]
    chars = plate_chars[longest_idx]

    return chars,info, plate_img

fig = plt.figure()
rows = 4
cols = 5
axes= []

MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.2, 1.0

thresh_row=[160,160,160,150,165,100,75,95,130,95,95,104,194,130,105,105,127,127,127]
path = 'C:/Users/82104/PycharmProjects/Inha_openCV/data/carNum/'
os.chdir(path)
files = os.listdir(path)

for i, src in enumerate(files):
    img = cv2.imread(path+src)
    img = imutils.resize(img, width=500 )
    imgH, imgW, imgC = img.shape


    # 이미지 전처리
    pre_img = preprocessing(img)

    # 이미지 후보군 찾기
    possible_dic, top30_contour_img, p_contour_img = possible_cont(pre_img)
    # 외곽선중에 번호판 후보군 추출
    result_idx = find_chars(possible_dic)

    # 최종 후보군 저장, 사각형 영역 그리기
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_dic, idx_list))

    temp_result_img = np.zeros((imgH, imgW, imgC), dtype=np.uint8)
    for r in matched_result:
        for d in r:
            cv2.rectangle(temp_result_img, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255,255,255), thickness=2)
    try:
        # 각도변경, 번호판 이미지 추출
        mPlate_img,info, rotate_img, cropped_img = angle(matched_result,thresh_row[i])

        # 전처리 후 문자인식
        final_char,final_info, final_plate_img = pytess(mPlate_img, info)
        print(final_char)

        # 최종으로 찾은 번호판 위치 표시
        final_img = rotate_img.copy()
        info = info[0]
        cv2.rectangle(final_img, pt1=(info['x'], info['y']), pt2=(info['x']+info['w'], info['y']+info['h']), color=(0,0,255), thickness=4)

        # 최종 이미지, 문자 저장
        axes.append(fig.add_subplot(rows,cols,i+1))
        plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        axes[-1].set_title(final_char)
        axes[-1].set_xticks([]), axes[-1].set_yticks([])
    except:
        print("번호판 인식 실패")
        axes.append( fig.add_subplot(rows,cols,i+1))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[-1].set_title('인식 실패')
        axes[-1].set_xticks([]), axes[-1].set_yticks([])

    if(i==0):
        cv2.imshow('img',img)
        cv2.waitKey()
        cv2.imshow('preprocessing',pre_img)
        cv2.waitKey()
        cv2.imshow('top30_contour_img',top30_contour_img)
        cv2.waitKey()
        cv2.imshow('possible_countour',p_contour_img)
        cv2.waitKey()
        cv2.imshow('matched_img',temp_result_img)
        cv2.waitKey()
        cv2.imshow('rotate_img',rotate_img)
        cv2.waitKey()
        cv2.imshow('cropped_img',cropped_img)
        cv2.waitKey()
        cv2.imshow('final_plate_img',final_plate_img)
        cv2.waitKey()
        cv2.imshow('final_img',final_img)
        cv2.waitKey()

plt.show()

