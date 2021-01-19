import matplotlib.pyplot as plt
import cv2

img_color_BGR = cv2.imread('Unknown-2.jpg')
img_color_RBG = cv2.cvtColor(img_color_BGR,cv2.COLOR_BGR2RGB)
plt.axis('off')
plt.imshow(img_color_RBG)
plt.show()

img_gray = cv2.imread('Unknown-2.jpg', cv2.IMREAD_GRAYSCALE)
plt.axis('off')
plt.imshow(img_gray, cmap='gray')
plt.show()

#두개의 이미지 한번에 출력하기
plt.subplot(121), plt.axis('off'), plt.imshow(img_color_RBG)
plt.subplot(122), plt.axis('off'), plt.imshow(img_gray,cmap='gray')
plt.show()