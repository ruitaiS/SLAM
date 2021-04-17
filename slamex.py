#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Initialize ORB keypoint detector
orb = cv2.ORB_create()

#Initialize Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#Create list of matching keypoints between frames
#Return list sorted by distance
def kpMatch (kp1, kp2):
    matches = bf.match(kp1.des, kp2.des)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches


#Use ORB to process the current frame for a (keypoint, descriptor) tuple
#Note that kp and des are arrays of keypoints and descriptors, not single elements
def process_frame (in_frame):
    kp, des = orb.detectAndCompute(in_frame, None)

    print(len(kp))

    #Calculate coordinate pairs for each keypoint, and draw a circle there
    for p in kp:
        coord = tuple(np.rint(list(p.pt)).astype(int))
        cv2.circle(in_frame, coord, 3, (255,0,0), -1)

    cv2.imshow("Frame", in_frame)

if __name__ == "__main__":
    cap = cv2.VideoCapture('foot.mp4')

    #Stores (keypoints, descriptors) tuple from past frames
    hold = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
            
            #waitkey(n) needs to be called to display anything
            #n displays the frame for n milliseconds
            #n = 0 pauses until keypress; n = 1 plays it normal-ish
            cv2.waitKey(1)            
        else:
            cap.release()
            break




