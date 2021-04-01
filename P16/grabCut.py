import cv2
import sys
import numpy as np

# cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode=None) -> mask, bgdModel, fgdModel
# img: 입력 영상. 8비트 3채널.
# mask: 입출력 마스크. cv2.GC_BGD(0), cv2.GC_FGD(1), cv2.GC_PR_BGD(2), cv2.GC_PR_FGD(3) 네 개의 값으로 구성됨. cv2.GC_INIT_WITH_RECT 모드로 초기화.
# rect: ROI 영역. cv2.GC_INIT_WITH_RECT 모드에서만 사용됨
# bgdModel: 임시 배경 모델 행렬. 같은 영상 처리 시에는 변경 금지.
# fgdModel: 임시 전경 모델 행렬. 같은 영상 처리 시에는 변경 금지.
# iterCount: 결과 생성을 위한 반복 횟수.
# mode: cv2.GC_로 시작하는 모드 상수. 보통 cv2.GC_INIT_WITH_RECT 모드로 초기화하고, cv2.GC_INIT_WITH_MASK 모드로 업데이트함.

#입력 영상 불러오기
src = cv2.imread('image.jpg')

if src is None:
    print('Image load failed')
    sys.exit()

# 사장형 지정을 통한 기초 분할
rc = cv2.selectROI(src)
mask = np.zeros(src.shape[:2], np.unit8) # 마스크는 검정색으로 채워져있고 입력 영상과 동일한 크기

# 결과를 계속 업데이트 하고 있으면 bgd, fgd 입력
cv2.grabCut(src, mask, rc, None, None, 5, cv2.GC_INIT_WITH_RECT)

# grabCut 자료에서 0,2는 배경, 1,3은 전경
# mask == 0 or mask == 2를 만족하면 0으로 설정 아니면 1로 설정
mask2 = np.where((mask == 0) | (mask == 2), 0, 1).astype('unit8')

#np.newaxis로 차원 확장
dst = src * mask2[:,:, np.newaxis]

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()