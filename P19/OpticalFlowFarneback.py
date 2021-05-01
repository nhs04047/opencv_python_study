import sys
import cv2
import numpy as np

def draw_flow(img, flow, step=16)
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # 입력 영상의 컬러 영상 변환
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 직선 그리기
    cv2.polylines(vis, lines, 0, (0, 255, 255), lineType=cv2.LINE_AA)

    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 128, 255), -1, lineType=cv2.LINE_AA)

    return vis


cap = cv2.VideoCapture('vtest.avi')

if not cap.isOpened():
    print('Camera open failed!')
    sys.exit()

ret, frame1 = cap.read()

if not ret:
    print('frame read failed!')
    sys.exit()

gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame2 = cap.read()

    if not ret:
        print('frame read failed!')
        sys.exit()

    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 13, 3, 5, 1.1, 0)

    cv2.imshow('frame2', draw_flow(gray2, flow))
    if cv2.waitKey(20) == 27:
        break

    gray1 = gray2

cv2.destroyAllWindows()

##################################################################################################

# 밀집 옵티컬 플로우 계산 함수

# cv2.calcOpticalFlowFarneback(prev, next, flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags) -> flow

# • prev, nex: 이전 영상과 현재 영상. 그레이스케일 영상.
# • flow: (출력) 계산된 옵티컬플로우. np.ndarray. shape=(h, w, 2), dtype=np.float32.
# • pyr_scale: 피라미드 영상을 만들 때 축소 비율. (e.g.) 0.5
# • levels: 피라미드 영상 개수. (e.g.) 3
# • winsize: 평균 윈도우 크기. (e.g.) 13
# • iterations: 각 피라미드 레벨에서 알고리즘 반복 횟수. (e.g.) 10
# • poly_n: 다항식 확장을 위한 이웃 픽셀 크기. 보통 5 또는 7.
# • poly_sigma: 가우시안 표준편차. 보통 poly_n = 5이면 1.1, poly_n = 7이면 1.5.
# • flags: 0, cv2.OPTFLOW_USE_INITIAL_FLOW, cv2.OPTFLOW_FARNEBACK_GAUSSIAN.