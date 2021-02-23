import sys
import cv2

src = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

# 트랙바 사용
def on_threshold(pos):
    _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('dst', dst)

cv2.imshow('src', src)
cv2.namedWindow('dst')
cv2.createTraskbar('Treshold', 'dst', 0, 255, on_threshold) # 임계값 범위 0 - 255
cv2.setTrackbarPos('Treshold', 'dst', 128) # 임계값 초기값 128s

cv2.waitKey()
cv2.destroyWindow()
