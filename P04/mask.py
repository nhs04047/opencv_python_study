import cv2
import sys

src = cv2.imread('cat.webp')
src = cv2.resize(src, dsize=(0,0),fx=3,fy=3)

# 4채널 영상인 png파일을 불러올때 사용, 4채널은 투명도
logo = cv2.imread('opencv-logo.png', cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, dsize=(0,0),fx=0.25,fy=0.25)

if src is None or logo is None:
    print('Image load failed')
    sys.exit()

# mask는 알파 채널로 만든 마스 영상
# 그레이스케일이어야함
mask = logo[:,:,3]

logo = logo[:,:,:-1]

h,w = mask.shape[:2]

# 마스크 연산을 위해서는 src와 dst의 영상 크기가 같아야함
crop = src[10:10+h, 10:10+w]

maskimg_img = cv2.copyTo(logo,mask,crop)

cv2.imshow('src',logo)
cv2.imshow('mask', mask)
cv2.imshow('drc', crop)
cv2.imshow('maskimg', maskimg_img)

cv2.waitKey(0)
cv2.destroyAllWindows()