import sys
import cv2

# CrCb 살색 히스토그램 구하기
ref = cv2.imread('image.png',cv2.IMREAD_COLOR)
mask = cv2.imread('image_mask.bmp', cv2.IMREAD_GRAYSCALE)

if ref is None or mask is None:
    print('Image load failed')
    sys.exit()

# BGR -> YCrCb
ref_yrcb = cv2.cvtColor(ref,cv2.COLOR_BGR2YCrCb)

# 히스토그램 생성
channels = [1,2] # CrCb 속성만 이용
ranges = [0, 256, 0, 256] # Cr, Cb 범위 지정
hist = cv2.calcHist([ref_yrcb], channels, mask, [128,128], ranges) # 히스토그램 생성
hist_norm = cv2.normalize(cv2.log(hist + 1), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


# 입력 영상에 히스토그램 역투영 적용하기
src = cv2.imread('image2.png', cv2.IMREAD_COLOR)

if src is None:
    print('Image load failed')
    sys.exit()

# 히스토그램 역투영을 위한 BGR -> YCrCb 전환
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

# 히스토그램 역투영
backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)

cv2.imshow('src', src)
cv2.imshow('hist_norm', hist_norm)
cv2.imshow('backproj', backproj)
cv2.waitKey()

cv2.destroyAllWindows()