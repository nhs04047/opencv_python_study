import sys
import numpy as np
import cv2

# 관심영역을 모서리 네개로 선택하는 함수
def drawROI(img, corners): #corners는 네 모서리 좌표
    cpy = img.copy() #그림을 그릴 이미지 복사

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
# 마우스 이벤트 처리
def onMouse(event, x, y, flags, param):  # 외관상 5개 인자. flags는 키가 눌린 여부, param은 전송 데이터
    global srcQuad, dragSrc, pt0ld, src  # 전역 변수 갖고 옴

    # 왼쪽 마우스가 눌렸을 때
    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:  # 클릭한 점이 원 안에 있는지 확인
                dragSrc[i] = True
                pt0ld = (x, y)  # 마우스를 이동할때 모서리도 따라 움직이도록 설정
                break

    if event == cv2.EVENT_LBUTTONUP:  # 마우스를 땜
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:  # 마우스 왼쪽 버튼이 눌려 있을 때 모서리 움직임
        for i in range(4):
            if dragSrc[i]:  # dragSrc가 True일 때
                dx = x - pt0ld[0]  # 이전의 마우스 점에서 dx, dy만큼 이동
                dy = y - pt0ld[1]

                srcQuad[i] += (dx, dy)  # 이동한 만큼 더해줌

                cpy = drawROI(src, srcQuad)
                cv2.imshow('img', cpy)  # 수정된 좌표로 모서리 이동
                pt0ld = (x, y)  # 현재 점으로 설정

src = cv2.imread('image.jpg')

if src is None:
    print('Image open failed')
    sys.exit

# 영상의 크기
h, w = src.shape[:2]
dw = 500 # 똑바로 핀 영상의 가로 크기
dh = round(dw * 297 / 210) # A4용지 크기

# 모서리 점들의 좌표, 드래그 상태 여부
# 내가 선택하려는 모서리 점 4개를 저장하는 넘파이 행렬, 30은 임의로 초기점의 좌표를 설정
# 완전히 구석이 아니라 모서리를 클릭할 수 있도록 자리를 둠
srcQuad = np.array([[500, 500], [500, h - 300], [w - 300, h - 300], [w - 300, 500]], np.float32)  # 모서리 위치

# 반시계 방향으로 출력 방향의 위치
dstQuad = np.array([[0, 0], [0, dh - 1], [dw - 1, dh - 1], [dw - 1, 0]], np.float32)

# 4개의 점 중에서 현재 어떤 점을 드래고 하고 있나 상태를 저장, 점을 선택하면 True, 떼면 False
dragSrc = [False, False, False, False]

# 모서리점, 사각형 그리기
# src에 srcQuad좌표를 전송해서 화면에 나타냄
disp = drawROI(src, srcQuad)

cv2.imshow('img', disp)
cv2.setMouseCallback('img', onMouse)

while True:
    key = cv2.waitKey()
    if key == 13:  # enter키, 엔터키 누르면 투시 변환과 결과 영상 출력
        break
    elif key == 27:  # ESC 키 종료
        cv2.destroyWindow('img')
        sys.exit()

# 투시변환
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)  # 3X3 투시 변환 행렬 생성
dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)  # 가로 세로 크기는 자동

# 결과 영상 출력
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destoryAllWindows()

