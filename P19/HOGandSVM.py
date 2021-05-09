import cv2
import sys
import numpy as np

# 마우스로 그림을 그리기 위해 마우스 좌표 갖고오기
oldx, oldy = -1, -1


# 마우스 그림 그리는 인터페이스
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


# 학습 데이터 & 레이블 행렬 생성
digits = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)

if digits is None:
    print('Image load failed!')
    sys.exit()

# 입력 영상의 높이, 넓이
h, w = digits.shape[:2]

# HOGDescriptor 객체 생성
# 영상크기 : 20,20 셀 크기: 5,5 스트라이드: 5,5 방향 히스토그램 빈: 0
hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (5, 5), 9)

# hog.getDescriptorSize 확인
print('Descriptor Size:', hog.getDescriptorSize())

# 입력 영상 20x20로 분할
cells = [np.hsplit(row, w // 20) for row in np.vsplit(digits, h // 20)]
cells = np.array(cells)
cells = cells.reshape(-1, 20, 20)  # shape = (5000, 20, 20)

# 각각의 셀에 대해서 hogDescriptor 계산
desc = []
for img in cells:
    # 1행 324열을 desc에 차곡차곡 채워 넣기
    desc.append(hog.compute(img))

# train_desc 생성. desc는 5000행 324열
train_desc = np.array(desc)  # (5000, 324, 1)

# (5000,324,1) -> (5000,324)
train_desc = train_desc.squeeze().astype(np.float32)

# train_labels 생성
train_labels = np.repeat(np.arrange(10), len(train_desc) / 10)

print('train_desc.shape:', train_desc.shape)
print('train_labels.shape:', train_labels.shape)

# SVM 학습

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)

# trainAuto로 찾은 값
svm.setC(2.5)
svm.setGamma(0.50625)

svm.train(train_desc, cv2.ml.ROW_SAMPLE, train_labels)
svm.save('svmdigits.yml')  # 학습 결과 저장, 이 파일을 이용하면 학습을 하지 않고 predict를 할 수 있다.

# 사용자 입력 영상에 대해 예측

img = np.zeros((400, 400), np.uint8)

cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse)

while True:
    key = cv2.waitKey()

    if key == 27:
        break
    elif key == ord(' '):  # 스페이스바 누르면 작동
        test_image = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        test_desc = hog.compute(test_image).T  # svm.predict는 1행 1열로 입력되어야 하므로

        _, res = svm.predict(test_desc)
        # 결과값 출력
        print(int(res[0, 0]))

        img.fill(0)
        cv2.imshow('img', img)

cv2.destroyAllWindows()