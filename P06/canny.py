import sys
import cv2

src = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE) # 그레이 스케일 영상

if src is None:
    print('Image load failed!')
    sys.exit()

# 하단 임계값과 상단 임계값은 실험적으로 결정하기
dst = cv2.Canny(src, 50, 150)
# cv2.Canny(image, threshold1, threshold2, edges=None, apertureSize=None, L2gradient=None) -> edges
# image: 입력 영상
# threshold1: 하단 임계값
# threshold2: 상단 임계값
# edges: 에지 영상
# apertureSize: 소벨 연산을 위한 커널 크기. 기본값은 3
# L2gradient: True이면 L2 norm 사용, False이면 L1 norm 사용. 기본값은 False.

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()