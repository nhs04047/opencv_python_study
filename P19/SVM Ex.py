import numpy as np
import cv2

# 8개의 데이터 생성
trains = np.array([[150, 200], [200, 250],
                   [100, 250], [150, 300],
                   [350, 100], [400, 200],
                   [400, 300], [350, 400]], dtype=np.float32)

# 앞 4개는 0번 클래스 뒤 4개는 1번 클래스로 지정
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC) # c 파라미터
# svm.setKernel(cv2.ml.SVM_LINEAR) # Gamma 파라미터
svm.setKernel(cv2.ml.SVM_RBF) # Gamma 파라미터

# trainAuto 함수가 C, Gamma 값을 결정해줌
svm.trainAuto(trains, cv2.ml.ROW_SAMPLE, labels)
print('C:', svm.getC())
print('Gamma:', svm.getGamma())

# trainAuto 함수로 알아낸 값을 train 이용하면 더 빠르게 작동 가능
# svm.setC(2.5)
# svm.setGamma(0.00001)
# svm.train(trains, cv2.ml.ROW_SAMPLE, labels)

# 시각화를 위한 코드
w, h = 500, 500
img = np.zeros((h, w, 3), dtype=np.uint8)

# h와 w 모든 좌표에 대해서 1행 2열 test 샘플 만듬
# test 샘플을 predict에 입력
for y in range(h):
    for x in range(w):
        test = np.array([[x, y]], dtype=np.float32)
        _, res = svm.predict(test)
        ret = int(res[0, 0]) # test 샘플이 몇번 클래스인지에 대한 정보, float이므로 int변환

        # 0 번 클래스는 빨강색
        if ret == 0:
            img[y, x] = (128, 128, 255)  # Red
        # 1 번 클래스는 녹색
        else:
            img[y, x] = (128, 255, 128)  # Green

color = [(0, 0, 128), (0, 128, 0)]

# 빨강색과 녹색을 원으로 출력되도록
for i in range(trains.shape[0]):
    x = int(trains[i, 0])
    y = int(trains[i, 1])
    l = labels[i]

    cv2.circle(img, (x, y), 5, color[l], -1, cv2.LINE_AA)

cv2.imshow('svm', img)
cv2.waitKey()
cv2.destroyAllWindows()