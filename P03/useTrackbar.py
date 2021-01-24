import cv2
import numpy as np

def on_level_change(pos):
    global img

    value = pos * 16

    value = np.clip(value, 0, 255) # 256이상일때 강제로 255로 변경

    img[:] = value
    cv2.imshow('image', img)

img = np.zeros((480, 640), np.uint8)
cv2.namedWindow('image')

# 창이 생성된 이후에 호출해야 함
cv2.createTrackbar('level', 'image', 0, 16, on_level_change)

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()