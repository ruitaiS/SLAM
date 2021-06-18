#!/usr/bin/env python3

import sys
import cv2
import numpy as np
from extractor import FeatureExtractor
 
fe = FeatureExtractor()

def process_frame(frame):
    kps, des, matches = fe.extract(frame)

    for p in kps:
        u,v = map(lambda x: int(round(x)), p.pt)
        #print((u,v))
        cv2.circle(frame, (u,v), radius=3, color=(0,255,0), thickness=-1)


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
        print("Video Error; Defaulting Input to Webcam")
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
            cv2.imshow("image", frame)
            cv2.waitKey(1)
        else:
            cap.release()
            break
