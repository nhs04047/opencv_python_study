import cv2
import numpy as np

def show_img():
    cv2.imshow('image1', img_color)
    cv2.waitKey(0)

#contour base
img_color = cv2.imread('shape_img.png')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img_color, contours, 0, (0,255,0), 3)

cv2.imshow('image1',img_color)
cv2.waitKey(0)

img_color = cv2.imread('star.jpg')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
ret, img_binary = cv2.threshold(img_gray, 127, 255, 0)

# 영역크기
for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)
show_img()

# 근사화
for cnt in contours:
    epsilon =cv2.arcLength(cnt, True) * 0.02
    approx_poly = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img_color, [approx_poly], 0, (0,0,255), 1)
show_img()

# 무게중심
for cnt in contours:
    M = cv2.moments(cnt)

    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cv2.circle(img_color, (cx,cy), 10, (255,0,0), -1)
show_img()

# 경계 사각형
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img_color, [box], 0, (0, 0, 255), 1)
show_img()

#Convex Hull
for cnt in contours:
    hull = cv2.convexHull(cnt)
    cv2.drawContours(img_color, [hull], 0, (255,255,0))
show_img()

#Convexity Defects
for cnt in contours:
    hull = cv2.convexHull(cnt, returnPoints = False)
    defects = cv2.convexityDefects(cnt, hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        print(d)

        if d> 500:
            cv2.line(img_color,start, end, [255,0,255],5)
            cv2.circle(img_color,far,5,[0,0,255], -1)

        show_img()


cv2.destroyAllWindows()
cv2.waitKey(0)