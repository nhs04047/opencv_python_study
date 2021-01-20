import sys
import cv2

callingCard = cv2.imread('image1.jpg')

if callingCard is None:
    print('image load failed')
    sys.exit()

callingCard_gray = cv2.cvtColor(callingCard,cv2.COLOR_BGR2GRAY)
ret, callingCard_bin = cv2.threshold(callingCard_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

cv2.imshow('callingCard', callingCard)
cv2.imshow('callingCard_gray', callingCard_gray)
cv2.imshow('callingCard_bin', callingCard_bin)

cv2.waitKey(0)
cv2.destroyAllWindows()