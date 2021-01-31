from transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils

# 이미지 로드 및 비율 조정
src = cv2.imread("image.jpg")
ratio = src.shape[0] / 500.0
orig = src.copy()
src = imutils.resize(src, height = 500)

# 그래이스케일로 이미지 전환, 노이즈 제거, 모서리 찾기
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
src_gray = cv2.GaussianBlur(src_gray, (7, 7), 0)
edged = cv2.Canny(src_gray, 75, 200)

# 원본과 앳지 이미지 보이기
# cv2.imshow("Image", src)력
# cv2.imshow("Edged", edged)
# cv2.waitKey()
# cv2.destroyWindow()

# 윤곽선 찾기, 가장 큰 윤곽선 저장
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse=True)[:5]

# 윤곽선 근사값 찾기
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

# 윤곽선 찾기 확인
# cv2.drawContours(src, [screenCnt], -1, (0,0,255), 2)
# cv2.imshow("image", src)
# cv2.waitKey()
# cv2.destroyWindow()

# 워핑 수행
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

# 결과 출
cv2.imshow("image", imutils.resize(orig, height = 650))
cv2.imshow("Scanned", imutils.resize(warped, height = 650))
cv2.waitKey()
cv2.destroyWindow()









