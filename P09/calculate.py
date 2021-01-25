import sys
import cv2
import matplotlib.pyplot as plt

src1 = cv2.imread('image1.jpg')
src2 = cv2.imread('image2.jpg')

if src1 is None or src2 is None:
    print('Image load failed!')
    sys.exit()

dst_add = cv2.add(src1, src2, dtype=cv2.CV_8U)
# cv2.add(src1, src2, dst=None, mask=None, dtype=None) -> dst
# 덧셈 연산 - cv2.add
# src1: (입력) 첫 번째 영상 또는 스칼라
# src2: (입력) 두 번째 영상 또는 스칼라
# dst: (출력) 덧셈 연산의 결과 영상
# mask: 마스크 영상
# dtype: 출력 영상(dst)의 타입. (e.g.) cv2.CV_8U, cv2.CV_32F 등 (cv 자료형 타입으로 입력)

dst_AW = cv2.addWeighted(src1, 0.5, src2, 0.5, 0.0)
# 가중치 합, 평균 연산 - cv2.addWeighted
# cv2.addWeighted(src1, alpha, src2, beta, gamma, dst=None, dtype=None) -> dst
# src1: (입력) 첫 번째 영상
# alpha: 첫 번째 영상 가중치
# src2: 두 번째 영상. src1과 같은 크기 & 같은 타입
# beta: 두 번째 영상 가중치
# gamma: 결과 영상에 추가적으로 더할 값
# dst: 가중치 합 결과 영상
# dtype: 출력 영상(dst)의 타입

dst_sub = cv2.subtract(src1, src2)
# 뺄셈 연산 - cv2.subtract
# cv2.subtract(src1, src2, dst=None, mask=None, dtype=None) -> dst
# src1: (입력) 첫 번째 영상 또는 스칼라
# src2: (입력) 두 번째 영상 또는 스칼라
# dst: (출력) 뺄셈 연산의 결과 영상
# mask : 마스크 영상
# dtype : 출력 영상(dst)의 타입

dst_ab = cv2.absdiff(src1, src2)
# 차이 연산 - cv2.absdiff
# cv2.absdiff(src1, src2, dst=None) -> dst
# src1: (입력) 첫 번째 영상 또는 스칼라
# src2: (입력) 두 번째 영상 또는 스칼라
# dst: (출력) 차이 연산의 결과 영상

# 영상 출력
plt.subplot(231), plt.axis('off'), plt.imshow(src1, 'gray'), plt.title('src1')
plt.subplot(232), plt.axis('off'), plt.imshow(src2, 'gray'), plt.title('src2')
plt.subplot(233), plt.axis('off'), plt.imshow(dst_add, 'gray'), plt.title('add')
plt.subplot(234), plt.axis('off'), plt.imshow(dst_AW, 'gray'), plt.title('addWeighted')
plt.subplot(235), plt.axis('off'), plt.imshow(dst_sub, 'gray'), plt.title('subtract')
plt.subplot(236), plt.axis('off'), plt.imshow(dst_ab, 'gray'), plt.title('absdiff')
plt.show()

# @영상논리연산
# cv2.bitwise_and(src1, src2, dst=None, mask=None) -> dst
# cv2.bitwise_or(src1, src2, dst=None, mask=None) -> dst
# cv2.bitwise_xor(src1, src2, dst=None, mask=None) -> dst
# cv2.bitwise_not(src1, dst=None, mask=None) -> dst

# src1: 첫 번째 영상 또는 스칼라
# src2: 두 번째 영상 또는 스칼라
# dst: 출력 영상
# mask: 마스크 영상