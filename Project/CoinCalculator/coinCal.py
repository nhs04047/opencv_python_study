import sys
import cv2
import numpy as np

src = cv2.imread('image.jpg')

if src is None:
    print('Image open failed!')
    sys.exit()

gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

blr = cv2.GaussianBlur(gray, (0,0), 1)

circles = cv2.HoughCircles(blr, cv2.HOUGH_GRADIENT, 1, 50, param1 = 150, param2 = 40, minRadius = 20, maxRadius = 80)

sum_of_money = 0
dst = src.copy()
if circles is not None:
    for i in range(circles.shape[1]):
        cx, cy, radius = circles[0][i]
        cv2.circle(dst, (cx, cy), int(radius), (0,0,255), 2, cv2.LINE_AA)

        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius)
        radius = int(radius)

        crop = dst[y1:y2, x1:x2, :]
        ch, cw = crop.shape[:2]

        mask = np.zeros((ch,cw), np.unit8)
        cv2.circle(mask, (cw//2, ch//2), radius, 255, -1)

        hsv = cv2.cvtColor(crop, cv2.CO)
