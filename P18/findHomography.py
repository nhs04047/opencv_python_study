import sys
import cv2
import numpy as np

# 영상 불러오기
src1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
src2 = cv2.imread('image_in_scene.jpg', cv2.IMREAD_GRAYSCALE)

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
feature = cv2.KAZE_create()  # 기본값인 L2놈 이용
# feature = cv2.AKAZE_create()
# feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(src1, None)
kp2, desc2 = feature.detectAndCompute(src2, None)

# 특징점 매칭
matcher = cv2.BFMatcher_create()
matches = matcher.match(desc1, desc2)

# 좋은 매칭 결과 선별
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:80]

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))

# 호모그래피 계산
# DMatch 객체에서 queryIdx와 trainIdx를 받아와서 크기와 타입 변환하기
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)
pts2 = np.array([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2).astype(np.float32)

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)  # pts1과 pts2의 행렬 주의 (N,1,2)

# 호모그래피를 이용하여 기준 영상 영역 표시
dst = cv2.drawMatches(src1, kp1, src2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

(h, w) = src1.shape[:2]

# 입력 영상의 모서리 4점 좌표
corners1 = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2).astype(np.float32)

# 입력 영상에 호모그래피 H 행렬로 투시 변환
corners2 = cv2.perspectiveTransform(corners1, H)

# corners2는 입력 영상에 좌표가 표현되있으므로 입력영상의 넓이 만큼 쉬프트
corners2 = corners2 + np.float32([w, 0])

# 다각형 그리기
cv2.polylines(dst, [np.int32(corners2)], True, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()

######################################################################################

# cv2.findHomography(srcPoints, dstPoints, method=None, ransacReprojThreshold=None, mask=None, maxIters=None, confidence=None) -> retval, mask

# • srcPoints: 1번 이미지 특징점 좌표. numpy.ndarray. shape=(N, 1, 2). dtype=numpy.float32.

# • dstPoints: 2번 이미지 특징점 점 좌표. numpy.ndarray. shape=(N, 1, 2). dtype=numpy.float32.

# • method: 호모그래피 행렬 계산 방법. 0, LMEDS, RANSAC, RHO 중 선택. 기본값은 0이며, 이상치가 있을 경우 RANSAC, RHO 방법 권장.

# • ransacReprojThreshold: RANSAC 재투영 에러 허용치. 기본값은 3.

# • maxIters: RANSAC 최대 반복 횟수. 기본값은 2000.

# • retval: 호모그래피 행렬. numpy.ndarray. shape=(3, 3). dtype=numpy.float32.

# • mask: 출력 마스크 행렬. RANSAC, RHO 방법 사용 시 Inlier로 사용된 점들을 1로 표시한 행렬. numpy.ndarray. shape=(N, 1), dtype=uint8