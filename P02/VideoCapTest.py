import sys
import cv2


cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(Video.avi)

if not cap.isOpened():
    print('camera open failed')
    sys.exit()

while True:
    ret,frame = cap.read()

    if ret == False:
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(frame, 50, 150)

    cv2.imshow("Color", frame)
    cv2.imshow("Gray", frame_gray)
    cv2.imshow('edge',edge)

    if cv2.waitKey(1)&0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()