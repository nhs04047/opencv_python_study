import sys
import cv2
import numpy as np

src = cv2.imread('image.jpg')

if src is None:
    print('Image open failed!')