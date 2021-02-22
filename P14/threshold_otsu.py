import sys
import cv2

src = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed')
    sys.exit()

# or 연산자로 OTSU 인자 입력
# 반환값 2개, 1개는 OTSU 임계값(실수형), 1개는 dst영상
# cv2.THRESH_OTSU 만 입력해도 됌
th, dst = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print("otsu's threshold:", th)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyWindow()