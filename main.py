
import cv2
import numpy as np
import mediapipe as mp
import time
import autopy

# import tiktak


ti=time.time()

mp_draw=mp.solutions.drawing_utils
mp_hand=mp.solutions.hands
wcr,hcr=autopy.screen.size()
print(wcr,hcr)
smoothing=10
plx,ply=0,0

clocx,clocY=0,0
tipIds=[4,8,12,16,20]

video=cv2.VideoCapture(0)
wcm=640
hcm=480
frameR=100
video.set(3,wcm)
video.set(4,hcm)

with mp_hand.Hands(min_detection_confidence=0.5,
               min_tracking_confidence=0.5) as hands:
    while True:
       
        ret,image=video.read()
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image.flags.writeable=False
        results=hands.process(image)
        image.flags.writeable=True
        image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lmList=[]
        if results.multi_hand_landmarks:
            for hand_landmark in results.multi_hand_landmarks:
                myHands=results.multi_hand_landmarks[0]
                for id, lm in enumerate(myHands.landmark):
                    h,w,c=image.shape
                    cx,cy= int(lm.x*w), int(lm.y*h)
                    lmList.append([id,cx,cy])
                mp_draw.draw_landmarks(image, hand_landmark, mp_hand.HAND_CONNECTIONS)
        fingers=[]
        
        if len(lmList)!=0:
            if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1,5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            total=fingers.count(1)
            # print(total)
            # if total ==5 :
            #     cv2.addText("hello")
                
            if len(lmList)!=0:
                x1,y1=lmList[8][1:]
                x2,y2=lmList[12][1:]
                
            # if fingers[1]==1 and fingers[2]==0 and fingers[0]==1:
            #     x5,y5=lmList[4][1:]
            #     length=math.hypot(x5-x1,y5-y1)
            #     print(length)


            if fingers[1] ==1 and fingers[2]==0:
                # print (x1,y1)
                cv2.rectangle(image,(frameR,frameR),(wcm-frameR,hcm-frameR),(255,0,0))
                x3=np.interp(x1,(frameR,wcm-frameR),(0,wcr))
                y3=np.interp(y1,(frameR,hcm-frameR),(0,hcr))
                clocx=plx+(x3-plx)/smoothing
                clocY=ply+(y3-ply)/smoothing
                autopy.mouse.move(wcr-clocx,clocY)
                cv2.circle(image,(x1,y1),10,(0,100,0),cv2.FILLED)
                plx,ply=clocx,clocY
           
                    

                
            if fingers[1]==1 and fingers[2]==1:
                autopy.mouse.click()
            
            # if fingers[1]==1 and fingers[2]==1 and fingers[3]==1 and fingers[4]==1 :
            #     time.sleep(1.5)
            #     break
            
            cv2.putText(image,str(total),(500,100),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)
            cv2.imshow("Frame",image)
            k=cv2.waitKey(1)
            if k==ord('q'):
                 break
ta=time.time()-ti

print("program execution time:",ta)
video.release()
cv2.destroyAllWindows()