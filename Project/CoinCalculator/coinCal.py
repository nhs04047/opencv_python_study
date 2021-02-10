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

