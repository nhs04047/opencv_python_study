# 레이블링 함수 - cv2.connectedComponents

# cv2.connectedComponents(image, labels=None, connectivity=None, ltype=None) -> retval, labels

# • image: 8비트 1채널 영상
# • labels: 레이블 맵 행렬. 입력 영상과 같은 크기. numpy.ndarray.
# • connectivity: 4 또는 8. 기본값은 8.
# • ltype: labels 타입. cv2.CV_32S 또는 cv2.CV_16S. 기본값은 cv2.CV_32S.
# • retval: 객체 개수. N을 반환하면 [0, N-1]의 레이블이 존재 하며, 0은 배경을 의미. (실제 흰색 객체 개수는 N-1개)
# retval : 객체 갯수 + 1 (배경 포함)을 반환.
# labels : 레이블맵 행렬을 반환.

# 객체 정보를 함께 반환하는 레이블링 함수 - cv2.connectedComponentsWinthStats

# cv2.connectedComponentsWithStats(image, labels=None, stats=None, centroids=None, connectivity=None, ltype=None) -> retval, labels, stats, centroids

# • image: 8비트 1채널 영상
# • labels: 레이블 맵 행렬. 입력 영상과 같은 크기. numpy.ndarray.
# • stats: 각 객체의 바운딩 박스, 픽셀 개수 정보를 담은 행렬. numpy.ndarray. shape=(N, 5), dtype=numpy.int32.
# • centroids: 각 객체의 무게 중심 위치 정보를 담은 행렬 numpy.ndarray. shape=(N, 2), dtype=numpy.float64.
# • ltype: labels 행렬 타입. cv2.CV_32S 또는 cv2.CV_16S. 기본값은 cv2.CV_32S
# retval : 객체 수 + 1 (배경 포함)
# labels : 객체에 번호가 지정된 레이블 맵
# stats : N행5열, N은 객체 수 +1 이며 각각의 행은 번호가 지정된 객체를 의미, 5열에는 x,y,width,height, area 순으로 정보가 담겨있음, x, y는 좌측 상단 좌표를 의미하며 area는 면적, 픽셀의 수를 의미
# centroids : N행 2열, 2열에는 x,y 무게 중심 좌표가 입력되어 있음, 무게 중심 좌표는 픽셀의 x좌표를 다 더해서 갯수로 나눈 값임, y좌표도 동일

import sys
import cv2

src = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

if src is None:
    print('Image load failed!')
    sys.exit()

_, src_bin = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)

cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)

dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

for i in range(1, cnt):
    (x,y,w,h,area) = stats[i]

    if area < 20:
        continue

    cv2.rectangle(dst, (x,y,w,h), (0, 255, 255))

cv2.imshow('src', src)
cv2.imshow('src_bin', src_bin)
cv2.waitKey('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows() #
