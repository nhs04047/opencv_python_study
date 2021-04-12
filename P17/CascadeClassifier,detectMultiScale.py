import sys
import cv2

src = cv2.imread('image.jpg')

if src is None:
    print('Image load failed!')
    sys.exit()

# 객체 생성
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')

# 개체가 재대로 생성됬는지 확인
if classifier.empty():
    print('XML load failed!')
    sys.exit()

# 입력영상에서 얼굴을 검출
faces = classifier.detectMultiScale(src) # 스케일팩터를 1.2로 지정해도 잘 작동함 더 빨라짐

# 각각의 행마다 (x,y,w,h) 받아와서 사각형을 그리는 코드
for (x, y, w, h) in faces:
    cv2.rectangle(src, (x, y, w, h), (255, 0, 255), 2)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()

# --------------------------------------------------------------------------------

# cv2.CascadeClassifier( ) -> <CascadeClassifier object>
# cv2.CascadeClassifier(filename) -> <CascadeClassifier object>
# filename을 지정했으면 .load를 안해도 됩니다.

# cv2.CascadeClassifier.load(filename) -> retval
# • filename: XML 파일 이름
# • retval: 성공하면 True, 실패하면 False

# -----

# cv2.CascadeClassifier.detectMultiScale(image, scaleFactor=None, minNeighbors=None, flags=None, minSize=None, maxSize=None) -> result

# • image: 입력 영상 (cv2.CV_8U)
# • scaleFactor: 영상 축소 비율. 기본값은 1.1.
# • minNeighbors: 얼마나 많은 이웃 사각형이 검출되어야 최종 검출 영역으로 설정할지를 지정. 기본값은 3.
# • flags: (현재) 사용되지 않음
# • minSize: 최소 객체 크기. (w, h) 튜플.
# • maxSize: 최대 객체 크기. (w, h) 튜플.
# • result: 검출된 객체의 사각형 정보(x, y, w, h)를 담은 numpy.ndarray. shape=(N, 4). dtype=numpy.int32.