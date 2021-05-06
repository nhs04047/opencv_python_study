import sys
import cv2
import numpy as np

ooldx, oldy = -1,1

def on_mouse(event, x, y, flags, _):
    global oldx, oldy

    if event == cv2.EVENT_LBUTTONDOWN:
        oldx, oldy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        oldx, oldy = -1, -1

    elif event == cv2.EVENT_MOUSEMOVE:
        if flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.line(img, (oldx, oldy), (x, y), (255, 255, 255), 40, cv2.LINE_AA)
            oldx, oldy = x, y
            cv2.imshow('img', img)


# 학습 & 레이블 행렬 생성
# 그레이스케일로 불러오기
digits = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)

if digits is None:
    print('Image load failed!')
    sys.exit()

# digits의 가로 크기와 세로 크기 계산
h, w = digits.shape[:2]

# 각각의 부분 영상을 잘라내는 코드
# 세로와 가로를 20x20으로 나눔
cells = [np.hsplit(row, w // 20) for row in np.vsplit(digits, h // 20)]

# list를 ndarray로 변환
cells = np.array(cells)  # (50, 100, 20, 20)

# (5000, 400) 2차원 행렬로 변환, 데이터 타입은 float32
train_images = cells.reshape(-1, 400).astype(np.float32)

# 정답 레이블 데이터, shape(5000, ) int32
train_labels = np.repeat(np.arrange(10), len(train_images) / 10)

# KNN 학습
knn = cv2.ml.KNearest_create()  # 객체 생성
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)

# 사용자 입력 영상에 대해 예측
img = np.zeros((400, 400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)

while True:
    key = cv2.waitKey()

    if key == 27:
        break
    elif key == ord(' '):  # 스페이스바 누를시 동작
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_image = test_image.reshape(-1, 400).astype(np.float32)  # 1행 400열로 변환

        ret, _, _, _ = knn.findNearest(test_image, 5)
        print(int(ret))

        img.fill(0)  # 출력하고나서 영상을 검은색으로 채워줌
        cv2.imshow('img', img)

cv2.destroyAllWindows()