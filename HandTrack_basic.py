import mediapipe as mp

import cv2

import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingerTextScale = 0.5
fingerTextThickness = 1

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imageHeight, imageWidth, imageCenter = img.shape
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * imageWidth), int(lm.y * imageHeight)
                if id == 0:  # WRIST
                    cv2.circle(img, (cx, cy), 13, (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "0(WRIST)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (0, 0, 255), fingerTextThickness)
                elif id == 4:  # THUMB
                    cv2.circle(img, (cx, cy), 10, (0, 127, 255), cv2.FILLED)
                    cv2.putText(img, "4(THUMB)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (0, 127, 255), fingerTextThickness)
                elif id == 8:  # INDEX
                    cv2.circle(img, (cx, cy), 10, (0, 255, 126), cv2.FILLED)
                    cv2.putText(img, "8(INDEX)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (0, 255, 126), fingerTextThickness)
                elif id == 12:  # MIDDLE
                    cv2.circle(img, (cx, cy), 10, (255, 255, 86), cv2.FILLED)
                    cv2.putText(img, "12(MIDDLE)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (255, 255, 86), fingerTextThickness)
                elif id == 16:  # RING
                    cv2.circle(img, (cx, cy), 10, (255, 0, 127), cv2.FILLED)
                    cv2.putText(img, "16(RING)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (255, 0, 127), fingerTextThickness)
                elif id == 20:  # PINKY
                    cv2.circle(img, (cx, cy), 10, (86, 255, 255), cv2.FILLED)
                    cv2.putText(img, "20(PINKY)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (86, 255, 255), fingerTextThickness)
                else:
                    cv2.putText(img, str(id), (cx + 10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, fingerTextScale, (255, 255, 255), fingerTextThickness)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (0, int(imageHeight*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Cam", img)
    cv2.waitKey(1)