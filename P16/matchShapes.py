import sys
import cv2
import numpy as np

# 영상 불러오기
# 2장의 영상을 사용합니다.
# 객체가 검은색, 배경이 흰색이라 findcontour을 이용할 때 반전해서 이용해야 합니다.
# symbols 에서 spades를 찾아내는 것이 목적입니다.
obj = cv2.imread('spades.png', cv2.IMREAD_GRAYSCALE)  # 찾고 싶은거
src = cv2.imread('symbols.png', cv2.IMREAD_GRAYSCALE)  # 대상

if src is None or obj is None:
    print('Image load failed!')
    sys.exit()

# 객체 영상 외곽선 검출
# 이진화, 반전
_, obj_bin = cv2.threshold(obj, 128, 255, cv2.THRESH_BINARY_INV)

# 외곽선 검출, EXTERNAL은 바깥 윤곽선, APPROX_NONE는 모든 외곽선 좌표 구하기
obj_contours, _ = cv2.findContours(obj_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 외곽선 정보를 obj_pts에 저장 (객체 1개)
obj_pts = obj_contours[0]

# 입력 영상 분석
# src 영상에서 객체가 검은색이므로 반전해서 이진화
_, src_bin = cv2.threshold(src, 128, 255, cv2.THRESH_BINARY_INV)

# 모든 정보를 저장
contours, _ = cv2.findContours(src_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 결과 영상
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

# 입력 영상의 모든 객체 영역에 대해서 contours에 있는 각각의 점들의 집합을 검출
for pts in contours:
    if cv2.contourArea(pts) < 1000:  # 영역 계산, 노이즈 제거
        continue

    # 외곽선 사각형 좌표 받아서 그리기
    rc = cv2.boundingRect(pts)  # 외곽선 좌표 입력하면 사각형 좌표 반환
    cv2.rectangle(dst, rc, (255, 0, 0), 1)

    # 두 개의 외곽선 정보를 비교해서 모양을 비교하는 용도로 사용
    # 이미 변형이 생긴 모양을 비교할 때는 동작이 잘 안됌
    # 기본적으로 회전, 크기 변환, 대칭, 이동
    dist = cv2.matchSapes(obj_pts, pts, cv2.CONTOURS_MATCH_I3, 0)

    # distance 값을 화면에 출력
    # spades와 같은 모양은 0.1보다 작은값
    cv2.putText(dst, str(round(dist, 4)), (rc[0], rc[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    # 0.1 보다 작은 건 빨간생으로 사각형
    if dist < 0.1:
        cv2.rectangle(dst, rc, (0, 0, 255), 2)

cv2.imshow('obj', obj)
cv2.imshow('dst', dst)
cv2.waitKey(0)


#-----------------------
# cv2.matchSahpes(contour1,contour2,method,parameter)->retval

# - contour1 : 첫 번째 외곽선 또는 그레이스케일 영상 - findcontour로 검출한 외곽선 정보를 입력해도 됨
# - contoir2 : 두 번째 외곽선 또는 그레이스케일 영상 - findcontoir로 검출한 외곽선 정보를 입력해도 됨
# - method : 비교 방법 지정, cv2.CONTOURS_MATCH_|1, cv2.CONTOURS_MATCH_|2, cv2.CONTOURS_MATCH_|3 중 하나 사용.
# - parameter : 사용되지 않음. 0 지정
# - retval : 두 외곽선 또는 그레이스케일 영상 사이의 거리(distance)