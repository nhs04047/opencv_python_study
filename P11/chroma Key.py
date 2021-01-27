import sys
import cv2

# 녹색 배경 동영상 불러오기
cap1 = cv2.VideoCapture('video.mp4')

if not cap1.isOpened():
    print('video open failed!')
    sys.exit()

# 비 오는 배경 동영상 불러오기
cap2 = cv2.VideoCapture('background.mp4')

if not cap2.isOpened():
    print('video open failed!')
    sys.exit

# 두 동영상의 크기, FPS는 같다고 가정
w = round(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_cnt1 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
frame_cnt2 = round(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
fps = round(cap1.get(cv2.CAP_PROP_FPS))

# 프레임 간 시간 간격 설정
delay = int(1000 / fps)

# 합성 여부 플래그
do_composit = False  # False 면 합성을 안함. True면 크로마키 합성

# 전체 동영상 재생
while True:  # 무한 루프
    ret1, frame1 = cap1.read()  # 녹색 배경 영상 읽어오기

    if not ret1:  # 영상1 프레임이 끝나면 종료
        break

    # do_composit 플래그가 True일 때에만 합성
    if do_composit:
        ret2, frame2 = cap2.read()  # 비 오는 영상 읽어오기

        if not ret2:  # 영상 2 프레임이 끝나면 종료
            break

        # HSV 색 공간에서 녹색 영역을 검출하여 합성
        hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (50, 150, 0), (70, 255, 255))  # 영상, 최솟값, 최댓값
        cv2.copyTo(frame2, mask, frame1)

    cv2.imshow('frame', frame1)
    key = cv2.waitKey(delay)

    # 스페이스바를 누르면 do_composit 플래그를 변경
    if key == ord(' '):  # 스페이스바를 누르면 True로 변경
        do_composit = not do_composit
    elif key == 27:  # esc 누르면 영상 종료
        break

cap1.release()  # 사용자 자원 해제
cap2.release()
cv2.destroyAllWindows()