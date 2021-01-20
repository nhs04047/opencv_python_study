import cv2
import numpy as np
import matplotlib.pyplot as plt

# 직선 그리기
img = np.full((512, 512,3), 255, np.uint8)
img = cv2.line(img, (0,0), (255,255), (255,0,0), 10)

plt.imshow(img)
plt.show()

# 사각형 그리기
img = np.full((512, 512,3), 255, np.uint8)
img = cv2.rectangle(img, (20,20), (255, 255), (255, 0, 0), -1)

plt.imshow(img)
plt.show()

# 원 그리기
img = np.full((512, 512,3), 255, np.uint8)
img = cv2.circle(img, (255,255), 50, (0,255,0), 10)

plt.imshow(img)
plt.show()

# 다각형 그리기
img = np.full((512, 512,3), 255, np.uint8)
point = np.array([[250,150], [128,200], [470, 444], [400, 150], [290,111]])
img = cv2.polylines(img, [point], True, (255,0,0), 10)

plt.imshow(img)
plt.show()

#  텍스트 그리기
img = np.full((512, 512,3), 255, np.uint8)
img = cv2.putText(img, 'Hello CV',(60,300), cv2.FONT_ITALIC, 3,(255,0,0))

plt.imshow(img)
plt.show()

