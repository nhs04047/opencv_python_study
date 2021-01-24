import sys
import numpy as np
import cv2

oldx = oldy = -1
# 좌표 기본값 설정

def on_mouse(event, x, y, flags, param):
    # event - 마우스 동작 event
    # x,y - 창의 기준으로 왼쪽 위 끝이 (0,0)
    # flage - 마우스 이벤트가 발생할 때 키보드 또는 마우스 상태를 의미

    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN: # 마우스 윈쪽 버튼이 눌렸을 때
        oldx, oldy = x,y
        print('EVENT_LBUTTONDOWN : %d, %d' %(x,y))

    elif event == cv2.EVENT_LBUTTONUP: # 마우스 윈쪽 버튼을 땠을 때
        print('EVENT_LBUTTONUP : %d, %d' %(x,y))

    elif event == cv2.EVENT_MOUSEMOVE: # 마우스가 움직을 때
        if flags & cv2.EVENT_FLAG_LBUTTON:

            cv2.line(img, (oldx, oldy), (x,y), (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('draw line', img)
            oldx,oldy = x,y

# 흰색 배경의 영상 생성
img = np.ones((480, 640, 3), dtype = np.uint8) * 255

cv2.namedWindow('draw line')

# 마우스 입력, namedWindow of imshow가 실행되어 창이 떠이쓴ㄴ 상태에서만 사용 가능
# 마우스 이벤트가 발생하면 on_mouse 함수 실행
cv2.setMouseCallback('draw line', on_mouse, img)

cv2.imshow('draw line', img)
cv2.waitKey(0)

cv2.destroyAllWindows()

