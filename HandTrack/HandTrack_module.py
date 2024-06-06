import mediapipe as mp
import cv2
import time



class handDetector():
    def __init__(self, staticImageMode=False, maxHands=2, minDetectConf=0.5, minTrackConf=0.5):
        self.mode = staticImageMode
        self.maxHands = maxHands
        self.DetectConf = minDetectConf
        self.TrackConf = minTrackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.DetectConf, min_tracking_confidence=self.TrackConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, drawLandmarks=True, drawKeyPointTitles=True):
        imageHeight, imageWidth, imageCenter = img.shape
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if drawLandmarks:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                if drawKeyPointTitles:
                    for id, lm in enumerate(handLms.landmark):
                        cx, cy = int(lm.x * imageWidth), int(lm.y * imageHeight)
                        if id == 0:  # WRIST
                            cv2.circle(img, (cx, cy), 13, (0, 0, 255), cv2.FILLED)
                            cv2.putText(img, "0(WRIST)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        elif id == 4:  # THUMB
                            cv2.circle(img, (cx, cy), 10, (0, 127, 255), cv2.FILLED)
                            cv2.putText(img, "4(THUMB)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 127, 255), 1)
                        elif id == 8:  # INDEX
                            cv2.circle(img, (cx, cy), 10, (0, 255, 126), cv2.FILLED)
                            cv2.putText(img, "8(INDEX)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 126), 1)
                        elif id == 12:  # MIDDLE
                            cv2.circle(img, (cx, cy), 10, (255, 255, 86), cv2.FILLED)
                            cv2.putText(img, "12(MIDDLE)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 86), 1)
                        elif id == 16:  # RING
                            cv2.circle(img, (cx, cy), 10, (255, 0, 127), cv2.FILLED)
                            cv2.putText(img, "16(RING)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 127), 1)
                        elif id == 20:  # PINKY
                            cv2.circle(img, (cx, cy), 10, (86, 255, 255), cv2.FILLED)
                            cv2.putText(img, "20(PINKY)", (cx + 15, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (86, 255, 255), 1)
                        else:
                            cv2.putText(img, str(id), (cx + 10, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img

    def findPosition(self, img, handNum=0, draw=False):
        lmList = list()
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                height, width, center = img.shape
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
        
        return lmList



def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img, True, False)
        img = cv2.flip(img, 1)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cv2.putText(img, str(int(fps)), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Cam", img)
        cv2.waitKey(1)


if __name__ == "main":
    main()