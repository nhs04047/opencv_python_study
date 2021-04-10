import sys
import cv2
import numpy as np

# 입력 영상 & 템플릿 영상 불러오기
src = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
templ = cv2.imread('temp.jpg', cv2.IMREAD_GRAYSCALE)

if src is None or templ is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상 밝기 50증가, 가우시안 잡음(sigma=10) 추가하여 약간의 변형을 줌
noise = np.zeros(src.shape, np.int32)

# cv2.randn은 가우시안 형태의 랜덤 넘버를 지정, 노이즈 영상에 평균이 50 시그마 10인 노이즈 추가
cv2.randn(noise, 50, 10)

# 노이즈를 입력 영상에 더함, 원래 영상보다 50정도 밝아지고 시그마 10정도 변형
src = cv2.add(src, noise, dtype=cv2.CV_8UC3)

# 탬플릿 매칭 & 결과 분석
res = cv2.matchTemplate(src, templ, cv2.TM_CCOEFF_NORMED)  # 여기서 최댓값 찾기

# 최솟값 0, 최댓값 255 지정하여 결과값을 그레이스케일 영상으로 만들기
res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 최댓값을 찾아야하므로 minmaxloc 사용, min, max, min좌표, max좌표 반환
_, maxv, _, maxloc = cv2.minMaxLoc(res)

# 탬플릿에 해당하는 영상이 입력 영상에 없으면 고만고만한 값에서 가장 큰 값을 도출.
# 그래서 maxv를 임계값 0.7 or 0.6을 설정하여 템플릿 영상이 입력 영상에 존재하는지 파악
print('maxv : ', maxv)
print('maxloc : ', maxloc)

# 매칭 결과를 빨간색 사각형으로 표시
# maxv가 어느 값 이상이여야지 잘 찾았다고 간주할 수 있다.
th, tw = templ.shape[:2]
dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
cv2.rectangle(dst, maxloc, (maxloc[0] + tw, maxloc[1] + th), (0, 0, 255), 2)

# 결과 영상 화면 출력
cv2.imshow('res_norm', res_norm)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()