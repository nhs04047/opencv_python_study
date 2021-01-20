import sys
import cv2
import numpy as np

callingCard = cv2.imread('image1.jpg')

if callingCard is None:
    print('image load failed')
    sys.exit()

callingCard_gray = cv2.cvtColor(callingCard,cv2.COLOR_BGR2GRAY)
ret, callingCard_bin = cv2.threshold(callingCard_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

#cv2.imshow('callingCard', callingCard)
#cv2.imshow('callingCard_gray', callingCard_gray)
cv2.imshow('callingCard_bin', callingCard_bin)

contours, _ = cv2.findContours(callingCard_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print(contours)

for pts in contours:
    if cv2.contourArea(pts) < 1000:
        continue

    # 외각선 근사화
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True)*0.02,True)

    if len(approx) != 4: #사각형 판
        continue

    # 기하학적 변환 - Perspective Transform
    w, h = 250, 400
    callingCardQuad = np.array([[approx[0, 0, :]], [approx[1, 0, :]],
                                [approx[2, 0, :]], [approx[3, 0, :]]]).astype(np.float32)
    dftQuad = np.array([[0, h], [w, h],[w, 0],[0, 0]]).astype(np.float32)

    pers = cv2.getPerspectiveTransform(callingCardQuad,dftQuad)
    dst = cv2.warpPerspective(callingCard, pers, (w,h))

    cv2.polylines(callingCard, pts, True, (0,0,255))

cv2.imshow('callingCard', callingCard)
cv2.imshow('perspective img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()