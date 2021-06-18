#!/usr/bin/env python3

import cv2
import numpy as np
from extractor import FeatureExtractor


 
fe = FeatureExtractor()

def process_frame(frame):
    kps, des, matches = fe.gftt(frame)

    for p in kps:
        u,v = map(lambda x: int(round(x)), p.pt)
        #print((u,v))
        cv2.circle(frame, (u,v), radius=3, color=(0,255,0), thickness=-1)


if __name__ == "__main__" :
    #VideoCapture([Filename String]) for pre-recorded video
    #VideoCapture(0) pulls from integrated webcam
    #VideoCapture(1) pulls from drone feed if active
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
