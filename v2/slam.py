#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from extractor import FeatureExtractor
def process_frame(frame):

    matches = fe.extract(frame)
    if matches is None:
        return   

    for p1, p2 in matches:
        #u,v = map(lambda x: int(round(x)), p1)
        #s,t = map(lambda x: int(round(x)), p2)
   
        #Denormalize
        u,v = fe.denormalize(p1)
        s,t = fe.denormalize(p2)

        #Circles on Matched Points
        #cv2.circle(frame, (u,v), radius=3, color=(0,255,0), thickness=-1)
        #cv2.circle(frame, (s,t), radius=3, color=(0,0,255), thickness=-1)

        #Line Between Matched Points
        cv2.line(frame,(u,v),(s,t),(0,255,0),1)



if __name__ == "__main__" :
 


    #VideoCapture([Filename String]) for pre-recorded video
    #VideoCapture(0) pulls from integrated webcam
    #VideoCapture(1) pulls from drone feed if active
    #cap = cv2.VideoCapture(0)

    #Opens file if passed as parameter
    #Else tries to open drone feed (assumes index 1)
    #Defaults to webcam
    cap = None
    if len(sys.argv)>1:
        cap = cv2.VideoCapture(str(sys.argv[1]))
    else:
        cap = cv2.VideoCapture(1)

    if cap is None or not cap.isOpened():
        print("Video Error; Defaulting to Webcam")
        cap = cv2.VideoCapture(0)

    #Use first frame to init Feature Extractor
    #F is some kinda parameter in the intrinsic Matrix, not sure
    fe = FeatureExtractor(cap.read()[1], F=1)


   

    #Main Detection Loop
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
            cv2.imshow("image", frame)
            cv2.waitKey(1)
        else:
            cap.release()
            break
