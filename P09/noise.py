import sys
import cv2

src = cv2.imread('noise.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

dst = cv2.medianBlur(src, 3)
# cv2.medianBlur(src, ksize, dst=None) -> dst
# src : 입력 영상. 각 채널 별로 처리됨
# ksize : 커널 크기. 1보다 큰 홀수를 지정. 숫자 하나를 집어주면 됌
# dst : 출력 영상, src와 같은 크기, 같은 타입

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()