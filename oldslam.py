#!/usr/bin/env python3
import cv2
import numpy as np
from matplotlib import pyplot as plt

orb = cv2.ORB_create()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
def kpMatch (tup1, tup2): #(tup of (kp, des, frame) or (kp, des)
    matches = bf.match(tup1[1], tup2[1])
    matches = sorted(matches, key = lambda x:x.distance) #sorts according to dist according to documentaiton but honestly dunno what its doing AFA code goes

    if len(tup1) == 3:#test case; we were passed the whole img frame as well, draw the two-frame comparison
        img = cv2.drawMatches(tup1[2], tup1[0], tup2[2], tup2[0],matches[:10], tup2[2], (255,0,0), (0,255,0),[],flags=2)

        img = cv2.resize(img, (990, 270))
        cv2.imshow("Frame", img)

    return matches


def matchParser (curr, prev, d):#returns a list of pairs of coordinates of matched points from the previous to the current
#curr; prev = (kp, des, frame) //frame is optional & would still work without the third element in the tuple
#culls elements whose distances are larger than d

    matches = kpMatch(curr, prev)
    kp1 = prev[0]
    kp2 = curr[0]
    result = []

    while len(matches) > 0:
        match = matches.pop()
        
        pPrev = kp1[match.trainIdx].pt
        pCurr = kp2[match.queryIdx].pt
        if ((pCurr[0]-pPrev[0])**2 + (pCurr[1]-pPrev[1])**2) < d:
            result.append((pCurr, pPrev))
        #reverse; since we started w/ the matches @ the end, aka the worst ones
        result.reverse()
    return result


#other options for better processing:
#grid up the input frame & run processing on each grid
#cv2.goodFeaturesToTrack is also maybe worth looking @
#rn don't super want to muck with that

def draw_kp (in_frame, kp): #draws kp's onto the frame that's provided
    

    #video scaling ex:
    #in_frame = cv2.resize(in_frame, (495, 270))
    for p in kp:
        #print (tuple(np.rint(list(p.pt)).astype(int)))
        #by some obnoxiousity, cv2.circle will only take coords as an int tuple
        #while the keypoint p.pt gives coords as a float tuple
        #vvv hence this ridiculous line below vvv
        coord = tuple(np.rint(list(p.pt)).astype(int))

        #https://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html
        #cv2.circle(img, color, thickness, lineType)
        cv2.circle(in_frame, coord, 3, (255,0,0), -1)

    #For saving video
    #out_file.write(in_frame)



    cv2.imshow("Frame", in_frame)

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


    #code for outputting to a vid file:
    fourcc = cv2.VideoWriter_fourcc(* 'XVID')
    out_file = cv2.VideoWriter("post.mp4", fourcc, 30.0, (640,480))

    #array for holding (kp, des) tuple from last 2 (or more) frames
    hold_cap = 2
    hold = []

    while cap.isOpened():
        #.read() >> Grabs, decodes and returns the next video frame as an array
#(possibly) L*W, with each entry being a color for that pixel in the frame? that's my hunch anyway.
        ret, frame = cap.read()
        if ret == True:

            #get keypoints and put into the s
            kp, des = orb.detectAndCompute(frame, None)
            hold.insert(0, (kp, des)) 

            if len(hold) > hold_cap:
                hold.pop()#cull the last element if the hold gets full

            #once hold is filled, start running matching alg on the last two sets of KP's recently inserted            
            if len(hold) == hold_cap:
                pairsList = matchParser(hold[0], hold[1], 25)
                drawMatch(frame, pairsList)
                print (pairsList)


            #draw the KPs we most recently found
            #draw_kp(frame, kp)

            #this line is needed to actually display anything
            #waitkey(n) displays the frame supposedly for n milliseconds
            #waitkey(0) displays a single frame until a key is pressed
            cv2.waitKey(1)
        else:
            out_file.release()
            cap.release()
            break




