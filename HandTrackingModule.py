import cv2 as cv
import mediapipe as mp
import time


class handDetector(): 
    
    def __init__(self, static_image_mode = False, max_num_hands = 2,
                  min_detection_confidence = 0.5,
                  min_tracking_confidence = 0.5):
         self.mode = static_image_mode
         self.max_hands = max_num_hands
         self.detection_Con = min_detection_confidence
         self.track_Con = min_tracking_confidence
         
         self.mp_hands = mp.solutions.hands
         self.hands = self.mp_hands.Hands(static_image_mode=self.mode, 
                                          max_num_hands=self.max_hands, 
                                          min_detection_confidence=self.detection_Con,
                                          min_tracking_confidence=self.track_Con)
         self.mp_draw = mp.solutions.drawing_utils
         
    
    def find_hands(self, img, draw = True):
        
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        self.result = self.hands.process(imgRgb)
        
        if self.result.multi_hand_landmarks:
            for hand_lms in self.result.multi_hand_landmarks:
                if draw :
                    self.mp_draw.draw_landmarks(img, hand_lms, 
                                            self.mp_hands.HAND_CONNECTIONS)
        return img
    
    
    def find_position(self, img, hand_number = 0, draw = True):
        
        lmList = []
        
        if self.result.multi_hand_landmarks:    
            
            my_hand = self.result.multi_hand_landmarks[hand_number]
            
            for id_, lm_ in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm_.x * w), int(lm_.y * h)
                lmList.append([id_, cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        
        return lmList
    
    
def main():
    cap = cv.VideoCapture(0)
    pTime = 0
    cTime = 0
    detector = handDetector()
    while True:
        
        success, img = cap.read()
        
        img = detector.find_hands(img)
        lmList = detector.find_position(img, draw=False)
        if len(lmList) != 0:    
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, 
                   (255, 0, 255), 3)
        
        cv.imshow("Image", img)
        cv.waitKey(1)
        
        
if __name__ == '__main__':
    main()