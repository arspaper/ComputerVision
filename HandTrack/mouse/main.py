import mediapipe as mp
import cv2
import time
import HandMouse_module as htm


flipCam = True

cap = cv2.VideoCapture(0)
detector = htm.Hand(False, 1)
while True:
    success, img = cap.read()
    if flipCam:
        img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img)
    if len(lmList) != 0:
        print(lmList[8])
    cv2.imshow("Cam", img)
    cv2.waitKey(1)