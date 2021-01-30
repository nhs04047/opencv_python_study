import sys
import numpy as np
import cv2

# 관심영역을 모서리 네개로 선택하는 함수
def drawROI(img, corners): #corners는 네 모서리 좌표
    cpy = img.copy #그림을 그릴 이미지 복사

    edge = (192, 192, 255) # 모서리 색상
    line = (128, 128, 255) # 선 색상

    for pt in corners:
        cv2.circle(cpy, tuple(pt), 25, edge, -1, cv2.LINE_AA)

    # 모서리를 잇는 선, 점들의 좌표는 튜플
    cv2.line(cpy, tuple(corners[0]), tuple(corners[1]), line, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1]), tuple(corners[2]), line, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2]), tuple(corners[3]), line, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3]), tuple(corners[0]), line, 2, cv2.LINE_AA)

    # addWeighted를 이용하여 입력 영상과 cpy영상에 가중치를 적용하여 투명도 적용
    # 모서리와 선에 의한 영역 설정 방해를 줄임
    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp

# 마우스 이벤트 처리
def onMouse(event, x, y, flags, param): # flags는 키가 눌린 여부, param은 전송 데이터
    global srcQuad, dragSrc, pt0ld, src

    # 왼쪽 마우스를 눌렀을 때
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i]-(x,y)) < 25: # 클릭한 점이 원 안에 있는지 확인
                dragSrc[i]=True
                pt0ld = (x,y) # 마우스를 이동할 때 모서리도 따라 움직이도록 설정
                break

            if event == cv2.EVENT_LBUTTONUP: # 마우스를 땜
                for i in range(4):
                    dragSrc[i] = False

            if event == cv2.EVENT_MOUSEMOVE: # 마우스 왼쪽 버튼이 눌려 있을 때 모서리 움직임
                for i in range(4):
                    if dragSrc[i]: # dragSrc가 True일 때
                        dx = x - pt0ld[0] # 이전의 마우스 점에서 dx, dy만큼 이동
                        dy = y - pt0ld[1]

                        srcQuad[i] += (dx, dy) # 이동한 만큼 더해줌

                        cpy = drawROI(src, srcQuad)
                        cv2.imshow('img', cpy) # 수정된 좌표로 모서리 이동
                        pt0ld = (x, y) # 현재 점으로 설정
                        
src = cv2.imread('docu.jpg')

if src is None:
    print('Image open failed')
    sys.exit

# 영상의 크기
h, w = src.shape[:2]
dw = 500 # 똑바로 핀 영상의 가로 크기
dh = round(dw * 297 / 210) # A4용지 크기

