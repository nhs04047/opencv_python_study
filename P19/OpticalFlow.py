import sys
import cv2

src1 = cv2.imread('frame1.jpg')
src2 = cv2.imread('frame2.jpg')

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 그레이스케일로 변환
gray1 = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)

# 코너점 찾는 함수, 그레이스케일 영상만 입력 가능
pt1 = cv2.goodFeaturesToTrack(gray1, 50, 0.01, 10)

# 찾은 코너점 정보를 옵티컬플로우 함수에 입력
# src1, src2에서 움직임 정보를 찾아내고 pt1에 입력한 좌표가 어디로 이동했는지 파악
pt2, status, err = cv2.calcOpticalFlowPyrLK(src1, src2, pt1, None)

# 가중합으로 개체가 어느 정도 이동했는지 보기 위함
dst = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)

# pt1과 pt2를 화면에 표시
for i in range(pt2.shape[0]):
    if status[i, 0] == 0:  # status = 0인 것은 제외, 잘못 찾은 것을 의미
        continue

    cv2.circle(dst, tuple(pt1[i, 0]), 4, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(dst, tuple(pt2[i, 0]), 4, (0, 0, 255), 2, cv2.LINE_AA)

    # pt1과 pt2를 이어주는 선 그리기
    cv2.arrowedLine(dst, tuple(pt1[i, 0]), tuple(pt2[i, 0]), (0, 255, 0), 2)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

##########################################################################################

# 루카스-카나데 옵티컬플로우 계산 함수

# cv2.calcOpticalFlowPyrLK(prevImg, nextImg, prevPts, nextPts, status=None, err=None, winSize=None, maxLevel=None, criteria=None, flags=None, minEigThreshold=None) -> nextPts, status, err

# • prevImg, nextImg: 이전 프레임과 현재 프레임. 8비트 입력 영상.
# • prevPts: 이전 프레임에서 추적할 점들. numpy.ndarray. shape=(N, 1, 2), dtype=np.float32.
# • nextPts: (출력) prevPts 점들이 이동한 (현재 프레임) 좌표.
# • status: (출력) 점들의 매칭 상태. numpy.ndarray. shape=(N, 1), dtype=np.uint8. i번째 원소가 1이면 prevPts의 i번째 점이 nextPts의 i번째 점으로 이동.
# • err: 결과 오차 정보. numpy.ndarray. shape=(N, 1), dtype=np.float32.
# • winSize: 각 피라미드 레벨에서 검색할 윈도우 크기. 기본값은 (21, 21).
# • maxLevel: 최대 피라미드 레벨. 0이면 피라미드 사용 안 함. 기본값은 3.
# • criteria: (반복 알고리즘의) 종료 기준