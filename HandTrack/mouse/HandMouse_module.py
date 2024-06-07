import mediapipe as mp
import cv2
import time
import pyautogui

pyautogui.FAILSAFE = False



class Hand():
    def __init__(self, sensetivity=1, staticImageMode=False, maxHands=2, minDetectConf=0.5, minTrackConf=0.5):
        self.mode = staticImageMode
        self.maxHands = maxHands
        self.DetectConf = minDetectConf
        self.TrackConf = minTrackConf
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.DetectConf, min_tracking_confidence=self.TrackConf)
        self.mpDraw = mp.solutions.drawing_utils
        self.conditionStartTime = None
        self.prevWristPos = None
        self.sens = sensetivity

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
                lmList.append([cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 13, (255, 0, 255), cv2.FILLED)
        
        return lmList

    def getModeCondition(self, lmList):
        wristPos = lmList[0]
        indexPos = lmList[5]
        middlePos = lmList[9]
        ringPos = lmList[13]
        pinkyPos = lmList[17]
        if  wristPos[1] < indexPos[1] and\
            wristPos[1] < middlePos[1] and\
            wristPos[1] < ringPos[1] and\
            wristPos[1] < pinkyPos[1]:
                if self.conditionStartTime is None:
                    self.conditionStartTime = time.time()
                
                if time.time() - self.conditionStartTime > 1.5:
                    return True
    
        else:
            self.conditionStartTime = None
        
        return False

    def mouse(self, lmList):
        wristPos = lmList[0]
        thumbPos = lmList[4]
        indexPos = lmList[8]
        middlePos = lmList[12]
        ringPos = lmList[16]
        pinkyPos = lmList[20]
        if (middlePos[0] < ringPos[0] < pinkyPos[0] or middlePos[0] > ringPos[0] > pinkyPos[0]) and thumbPos[1] < middlePos[1]:  # 'Claw' mode
            if self.prevWristPos is None:
                self.prevWristPos = wristPos
                return
            xPos, yPos = pyautogui.position()
            xHandPos, yHandPos = wristPos
            xPrevHandPos, yPrevHandPos = self.prevWristPos
            dx, dy = xHandPos - xPrevHandPos, yHandPos - yPrevHandPos
            self.prevWristPos = wristPos
            if abs(dx) > 1 or abs(dy) > 1:
                pyautogui.moveTo(int(xPos + dx * self.sens), int(yPos + dy * self.sens))
            
            if ((indexPos[0] - thumbPos[0]) ** 2 + (indexPos[1] - thumbPos[1]) ** 2) ** 0.5 < 12:
                pyautogui.click()
        


        else:
            self.prevWristPos = None



def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = Hand()
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