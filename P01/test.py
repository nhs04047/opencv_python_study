import sys
import cv2


img_color = cv2.imread('Unknown-2.jpg', cv2.IMREAD_COLOR)

if img_color is None:
    print('Image lad failed!')
    sys.exit()

cv2.namedWindow('Show Image')
cv2.imshow('Show Image', img_color)

cv2.waitKey(0)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow('Show Image', img_gray)
cv2.waitKey(0)

cv2.destroyAllWindows()

