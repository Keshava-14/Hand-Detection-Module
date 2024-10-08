import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands

hands = mpHands.Hands()

mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    
    success, img = cap.read()
    
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    #print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id_, lm_ in enumerate(handLms.landmark):
                #print(id_, lm_)
                h, w, c = img.shape
                cx, cy = int(lm_.x * w), int(lm_.y * h)
                print(id_, cx, cy)
                if id_ == 0:
                    cv.circle(img, (cx, cy), 25, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    
    pTime = cTime
    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, 
               (255, 0, 255), 3)
    
    cv.imshow("Image", img)
    cv.waitKey(1)