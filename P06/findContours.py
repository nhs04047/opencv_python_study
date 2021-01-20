import sys
import cv2

callingCard = cv2.imread('image1.jpg')

if callingCard is None:
    print('image load failed')
    sys.exit()

callingCard_gray = cv2.cvtColor(callingCard,cv2.COLOR_BGR2GRAY)
ret, callingCard_bin = cv2.threshold(callingCard_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#cv2.imshow('callingCard', callingCard)
#cv2.imshow('callingCard_gray', callingCard_gray)
cv2.imshow('callingCard_bin', callingCard_bin)

contours, _ = cv2.findContours(callingCard_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(contours)

#외각선 근사화
for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02,True)

    if len(approx) != 4:
        continue

    cv2.polylines(callingCard, pts, True, (0,0,255))


cv2.waitKey(0)
cv2.destroyAllWindows()