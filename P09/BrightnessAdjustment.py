import sys
import cv2
import numpy as np

#  그레이스케일 영상 밝기 조절
src = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# add함수 이용
dst = cv2.add(src, 100)
# cv2.add(src1, src2, dst=None, mask=None, dtype=None) -> dst
# src1 - (입력) 첫 번째 영상 또는 스칼라
# src2 - (입력) 두 번째 영상 또는 스칼라
# dst - (출력) 덧셈연산의 결과 영상
# mask - 마스크 영상
# dtype - 출력 영상(dts)의 타입용

# numpy를 이용
dst = np.clip(src+100., 0, 255).astype(np.uint8)  # 0~255

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

# 컬러스케일 영상 밝기 조절
src = cv2.imread('image.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# add함수 이용
dst = cv2.add(src, (100, 100, 100, 0)) # 4채널

# numpy 이용
# dst = np.clip(src+100., 0, 255).astype(np.uint8)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()