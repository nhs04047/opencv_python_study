import sys
import cv2
import numpy as np

src = cv2.imread('image.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상의 높이와 넓이 정보 추출
h, w = src.shape[:2]

# np.indice는 행렬의 인덱스값 x좌표값 y좌표값을 따로따로 행렬의 형태로 변환해줌
map2, map1 = np.indices((h, w), dtype=np.float32)

# y좌표에 sin함수를 줬는데 파도처럼 하기 위해서
# y좌표 값에 10픽셀만큼 꿀렁꿀렁 거릴 수 있도록.
# sin함수가 x좌표를 이용해서 파도를 만들기 위해 map1을 줌
# 적당한 값을 나눠서 여러번 꿀렁꿀렁 거리게
map2 = map2 + 10 * np.sin(map1 / 32)

# borderMode는 근방의 색깔로 대칭되게 해서 채워줌, 기본값은 빈 공간을 검은색으로 표현
dst = cv2.remap(src, map1, map2, cv2.INTER_CUBIC, borderMode=cv2.BORDER_DEFAULT)

cv2.imshow('src', src)
cv2.imshow('dst', dst)
cv2.waitKey()

cv2.destroyAllWindows()