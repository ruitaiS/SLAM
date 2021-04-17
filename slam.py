#!/usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

#takes (kp, des, frame) or (kp, des)
#if frame is passed, draws side by side compare
#raw output of bf.match;
#call matchParse to translate to coord pairs
def kpMatch (tup1, tup2): 
    matches = bf.match(tup1[1], tup2[1])
    return matches
'''
#put this in it's own function
    if len(tup1) == 3:
        img = cv2.drawMatches(tup1[2], tup1[0], tup2[2], tup2[0],matches[:10], tup2[2], (255,0,0), (0,255,0),[],flags=2)
        img = cv2.resize(img, (990, 270))
        cv2.imshow("Frame", img)
'''

#culls pairsList based on func : pair -> bool
def matchCull (pairsList, func):
    outList = []
    for pair in pairsList:
        if func (pair):
            outList.append(pair)
    return outList

#specify max square of distance btwn points
#:int -> (pair -> bool)
def dSquare (d):
    def retFunc (pair):
        p1 = pair[0]
        p2 = pair[1]
        return ( ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) < d )
    return retFunc
    
def passThrough (pair):
    return True

#def inNSD (pairlist, n):
    #takes pairList & desired # of max stdevs
    #creates function (pair) that tells t/f
    
#curr, prev are (kp, des, frame) or (kp, des)  tuples
#returns list of coord pairs
def matchParse (prev, curr):
    #definitely broke some shit in here

    matchList = kpMatch(prev, curr)

    kp1 = prev[0]
    kp2 = curr[0]
    pairsList = []

    for match in matchList:
        pPrev = kp1[match.trainIdx].pt
        pCurr = kp2[match.queryIdx].pt
        pairsList.append((pCurr, pPrev))
    return pairsList

#draws kp's onto the frame that's provided
def drawKP (in_frame, kp): 
    #scaling:
    #in_frame = cv2.resize(in_frame, (495, 270))
    for p in kp:
        #print (tuple(np.rint(list(p.pt)).astype(int)))

        #keypoint p.pt gives coords as a float tuple
        #cv2.circle takes int tuple
        coord = tuple(np.rint(list(p.pt)).astype(int))

        #https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
        cv2.circle(in_frame, coord, 3, (255,0,0), -1)

    #Saving video
    #out_file.write(in_frame)
    cv2.imshow("Frame", in_frame)

#draws lines for coords between matches
#from kp in last frame to kp in curr frame
def drawMatch (frame, pairsList):
    for i in pairsList:
        a = tuple(np.rint(list(i[0])).astype(int))
        b = tuple(np.rint(list(i[1])).astype(int))
        cv2.line(frame, a, b,(255,0,0),5)
    cv2.imshow("hi", frame)

if __name__ == "__main__":
    #https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html
    #cap = cv2.VideoCapture('foot.mp4')
    cap = cv2.VideoCapture('fpv.avi')

    #Outputting vid to File:
    #fourcc = cv2.VideoWriter_fourcc(* 'XVID')
    #out_file = cv2.VideoWriter("post.mp4", fourcc, 30.0, (640,480))

    #array for holding (kp, des) tuple from last 2 (or more) frames
    hold_cap = 2
    hold = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            #stock up the hold
            #reminder - look up cv2.ORB() class function
            kp, des = orb.detectAndCompute(frame, None)
            
            #(kp, des, frame) also accepted
            #newest first
            hold.insert(0, (kp, des)) 

            if len(hold) > hold_cap:
                #cull oldest if full
                hold.pop()

            #run matching once hold filled            
            if len(hold) == hold_cap:
                #reminder - seperate matchParse from kp_match
                #as-is matchParse *calls* and parses return from kp_match
                pairsList = matchCull (matchParse(hold[0], hold[1]), dSquare(25))
                drawMatch(frame, pairsList)
                #print (pairsList)

            #Draw kps (possibly made obsolete by drawMatch)
            #drawKP(frame, kp)

            #waitkey(n) displays the frame supposedly for n milliseconds
            #waitkey(0) displays a single frame until a key is pressed
            cv2.waitKey(1)
        else:#no more frames; release cap & quit
            #out_file.release()
            cap.release()
            break

'''
other options for better processing:
grid up the input frame & run processing on each grid
cv2.goodFeaturesToTrack is also maybe worth looking @

To-Do
-kp_match & matchParse:
mP actually calls KP
KP contains code to draw
These two are kind of a mess tbh

-drawKP might be obsolete / incompatible with drawMatch

-modularize video saving (rn. it's commented out)

-read up on openCV orb doc

'''
