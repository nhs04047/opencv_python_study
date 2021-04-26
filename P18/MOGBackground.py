import sys
import cv2
# 비디오 파일 열기
cap = cv2.VideoCapture('Video.avi')

if not cap.isOpened():
    print('Video open failed!')
    sys.exit()

# 배경 차분 알고리즘 객체 생성
bs = cv2.createBackgroundSubtractorMOG2()
#bs = cv2.createBackgroundSubtractorKNN() # 배경영상이 업데이트 되는 형태가 다름
#bs.setDetectShadows(False) # 그림자 검출 안하면 0과 255로 구성된 마스크 출력

# 비디오 매 프레임 처리
while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 0또는 128또는 255로 구성된 fgmask 생성
    fgmask = bs.apply(gray)
    # 배경 영상 받아오기
    back = bs.getBackgroundImage()

    # 레이블링을 이용하여 바운딩 박스 표시
    cnt, _, stats, _ = cv2.connectedComponentsWithStats(fgmask)

    for i in range(1, cnt):
        x, y, w, h, s = stats[i]

        if s < 80:
            continue

        cv2.rectangle(frame, (x, y, w, h), (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('back', back)
    cv2.imshow('fgmask', fgmask)

    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()

########################################################################################

# BackgroundSubtractorMOG2클래스 생성 함수
# cv2.createBackgroundSubtractorMOG2(, history=None, varThreshold=None, detectShadows=None) -> dst
# • history: 히스토리 길이. 기본값은 500.
# • varThreshold: 픽셀과 모델 사이의 마할라노비스 거리(Mahalanobis distance) 제곱에 대한 임계값. 해당 픽셀이 배경 모델에 의해 잘 표현되는 지를 판단. 기본값은 16.
# • detecShadows: 그림자 검출 여부. 기본값은 True

# 전면 객체 마스크 생성 함수
# cv2.BackgroundSubtractor.apply(image, fgmask=None, learningRate=None) -> fgmask
# • image: (입력) 다음 비디오 프레임
# • fgmask: (출력) 전경 마스크 영상. 8비트 이진 영상.
# • learningRate: 배경 모델 학습 속도 지정 (0~1 사이의 실수). 기본값은 -1.
 # 0은 배경 모델을 갱신하지 않음
 # 1은 매 프레임마다 배경 모델을 새로 만듦
 # -1은 자동으로 결정됨

# 배경 영상 반환 함수
# cv2.BackgroundSubtractor.getBackgroundImage(, backgroundImage=None) -> backgroundImage
# • backgroundImage: (출력) 학습된 배경 영상