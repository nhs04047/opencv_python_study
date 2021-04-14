import sys
import cv2
import numpy as np

# 영상 불러오기
src1 = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
feature = cv2.KAZE_create()  # 방향 성분은 표현이 안됌
# feature = cv2.AKAZE_create() # 카제를 빠르게, accelateKaze, 방향선분 표현
# feature = cv2.ORB_create() # 가장 빠르지만 성능이 떨어짐

# 특징점 검출
kp1 = feature.detect(src1)
kp2 = feature.detect(src2)

# 검출된 특징점 갯수 파악
print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))

# 검출된 특징점 출력 영상 생성
dst1 = cv2.drawKeypoints(src1, kp1, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst2 = cv2.drawKeypoints(src2, kp2, None,
                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 영상 출력
cv2.imshow('dst1', dst1)
cv2.imshow('dst2', dst2)
cv2.waitKey()
cv2.destroyAllWindows()

