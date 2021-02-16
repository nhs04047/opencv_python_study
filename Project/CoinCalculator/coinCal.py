import sys
import cv2
import numpy as np

# 입력 이미지 불러오기
src = cv2.imread('image.png')

if src is None:
    print('Image open failed!')
    sys.exit()

# 흑백 영상으로 변환(원을 검출하기 위해)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# 허프 변환 함수가 노이즈에 민감하기 때문에 가우시안 블러로 노이즈 제거
blr = cv2.GaussianBlur(gray, (0, 0), 1)

# 허프 변환 원 검출
# 트랙바를 이용한 테스트로 파라미터 값 결정
circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50,
                           param1=150, param2=40, minRadius=20, maxRadius=80)

# 원 검출 결과 및 동전 금액 출력
sum_of_money = 0
dst = src.copy()
if circles is not None:  # 원이 검출 됬으면
    for i in range(circles.shape[1]):  # 원의 개수 만큼 반복문
        cx, cy, radius = circles[0][i]  # 중심좌표, 반지름 정보 얻기
        cv2.circle(dst, (cx, cy), int(radius), (0, 0, 255), 2, cv2.LINE_AA)  # 얻은 정보로 원 그리기

        # 동전 영역 부분 크롭 영상 만들기
        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius)
        radius = int(radius)

        crop = dst[y1:y2, x1:x2, :]  # 크롭 영상 생성
        ch, cw = crop.shape[:2]  # 크롭 영상의 세로, 가로 정보 획득

        # 동전 영역에 대한 ROI 마스크 영상 생성, 배경을 없애기 위함
        mask = np.zeros((ch, cw), np.uint8)  # 크롭 영상 크기와 동일한 검정색 마스크
        cv2.circle(mask, (cw // 2, ch // 2), radius, 255, -1)  # 검출된 원크기의 흰색원 그림

        # 동전 영역 Hue 색 성분을 +40 시프트하고, Hue 평균을 계산
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue, _, _ = cv2.split(hsv)  # hsv 정보 받아오기
        hue_shift = (hue + 40) % 180  # 나머지 연산을 통해 180 초과하면 0으로
        mean_of_hue = cv2.mean(hue_shift, mask)[0]  # 마스크 범위만 hue 평균 계산하기

        # Hue 평균이 90보다 작으면 10원, 90보다 크면 100원으로 간주
        won = 100
        if mean_of_hue < 90:
            won = 10

        sum_of_money += won

        cv2.putText(crop, str(won), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 0, 0), 2, cv2.LINE_AA)

cv2.putText(dst, str(sum_of_money) + 'won', (40, 80),
            cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()