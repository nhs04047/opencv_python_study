import sys
import cv2
import numpy as np

# 입력 영상 불러오기
# 640x480
src = cv2.imread('flowers.jpg')

if src is None:
    print('Image load failed')
    sys.exit()

# 차원 변환 $ np.float32 자료형 변환
# 307200, 3
data = src.reshape((-1, 3)).astype(np.float32)

# K-means 알고리즘
# 최대 10번 반복하고 1픽셀 이하로 움직이면 종료
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

for K in range(5, 11):
    print('K:', K)
    # label은 각각의 데이터가 속한 군집 정보, center은 군집의 중심점 좌표
    ret, label, center = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 군집화 결과를 이용하여 출력 영상 생성
    center = np.unit8(center)

    # 중심점 좌표를 받아서 dst에 입력 (307200, 3) 3은 중심 좌표
    dst = center[label.flatten()]  # 각 픽셀을 K개 군집 중심 색상으로 치환

    # 입력 영상과 동일한 형태로 변환 (640,480,3)
    dst = dst.reshape(src.shape)

    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()

cv2.destroyAllWindows()