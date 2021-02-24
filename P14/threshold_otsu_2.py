import sys
import cv2
import numpy as np

src = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load faild!')
    sys.exit()

# 전역 이진화
_, dst1 = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 지역 이진화
dst2 = np.zeros(src.shape, np.unit8)

# 넓이 / 4, 높이 / 4
bw = src.shape[1]//4
bh = src.shape[0]//4

# 가로 세로 4등분 하기
for y in range(4):
    for x in range(4):
        src_ = src[y*bh : (y+1)*bh. x*bw:(x+1)*bw] # threshold 입력값으로 주기 위해 입력 영상도 등분
        dst_ = dst2[y*bh : (y+1)*bh. x*bw:(x+1)*bw] # dst_를 변경하면 dst2도 변경됨

        cv2.threshold(src_, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst_)

cv2.imshow('src', src)
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyWindow()