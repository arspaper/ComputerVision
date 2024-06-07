import mediapipe as mp
import cv2
import HandMouse_module as htm



flipCam = True
sensetivity = 1



cap = cv2.VideoCapture(0)
Hand = htm.Hand(1, False, sensetivity)

mouseConnection = False
stillInConnectionZone = False

while True:
    success, img = cap.read()
    if not success:
        break
    if flipCam:
        img = cv2.flip(img, 1)
    
    img = cv2.resize(img, (640, 480))

    img = Hand.findHands(img, True, False)

    lmList = Hand.findPosition(img)

    if len(lmList) != 0:  # checks if any positions were given
        if Hand.getModeCondition(lmList) == True:  # if hand is in change mode zone
            if stillInConnectionZone is False and mouseConnection is False:
                mouseConnection = True
                stillInConnectionZone = True
            if stillInConnectionZone is False and mouseConnection is True:
                mouseConnection = False
                stillInConnectionZone = True
        else:
            stillInConnectionZone = False
        
        if mouseConnection:  # mouse is 'connected' to the hand
            Hand.mouse(lmList)

    if mouseConnection:
        colorConnection = (0, 255, 0)
    else:
        colorConnection = (0, 0, 255)
    cv2.putText(img, str(f"Mouse: {mouseConnection}"), (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, colorConnection, 2)
    cv2.putText(img, str(f"Sens: {sensetivity}"), (0, 48), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("HandMouse", img)
    cv2.waitKey(1)
    if cv2.getWindowProperty("HandMouse", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()