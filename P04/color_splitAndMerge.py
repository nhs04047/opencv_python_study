import sys
import cv2

src = cv2.imread('image.jpg')

if src is None:
    print('image load failed')
    sys.exit()

# 영상 색상 속성 확인
print('image shape : ', src.shape)
print('image type : ', src.dtype)

# BGR 색 영역 분할 - cv2.split()
b_plane, g_plane, r_plane = cv2.split(src)

# 슬라이싱을 이용한 BGR 색 평면 분할
# b_plane = src[:,:,0]
# g_plane = src[:,:,1]
# r_plane = src[:,:,2]

cv2.imshow('src', src)
cv2.imshow('Blue', b_plane)
cv2.imshow('Green', g_plane)
cv2.imshow('Red', r_plane)
cv2.waitKey()

# 색상결합 - cv2.merge()
src_merge = cv2.merge((b_plane,g_plane,r_plane))

cv2.imshow('src', src)
cv2.imshow('color merge', src_merge)
cv2.waitKey()

cv2.destroyAllWindows()

