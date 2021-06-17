#!/usr/bin/env python3

import cv2
import numpy as np

class FeatureExtractor(object):

    #Contains different methods for extracting keypoints from an img

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures = 1000)
        self.bf = cv2.BFMatcher()
        self.last = None

    #Subdivides image into grid and extracts keypoints from each
    def gridExtract(self, img):
        GX = 100 # X dim of grid square
        GY = 100 # Y dim of grid square
        rkp = [] #returned keypoints

        print("Img X: " + str(img.shape[0]) + "Img Y: " + str(img.shape[1]))

        for ry in range(0, img.shape[0], GY):
            for rx in range(0, img.shape[1], GX):
                gridSection = img[ry:ry+GY, rx:rx+GX]
                #print("Grid X Origin: " + str(rx) + "Grid Y Origin: " + str(ry))
                #print("Grid X: " + str(gridSection.shape[0]) + "Grid Y: " + str(gridSection.shape[1]))

                kp = self.orb.detect(gridSection, None)
                #print(len(kp))
                for p in kp:
                    #offset by grid section origin coords
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    rkp.append(p)
        return rkp

    def extract(self, img):
        print(type(img))
        kp = self.orb.detect(img, None)
        return kp

    def gftt(self, img):    
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance = 3)

        if feats is not None:
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        else:
            kps = []
        des = self.orb.compute(img, kps)

        #self.last = {'kps': kps, 'des': des} #not totally sure what this line does
        #if self.last is not None:
        #    matches = self.bf.match(des, self.last['kps'])
        #    print(matches)

        return kps, des
 
fe = FeatureExtractor()

def process_frame(frame):

    #detectAndCompute() -> list of keypoints, list of descriptors
    #https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

    #keypoints:
    #https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

    #kp.pt is the coordinates of the keypoint (type Point2f)
    #has tuple 
    #https://docs.opencv.org/3.4/dc/d84/group__core__basic.html#ga7d080aa40de011e4410bca63385ffe2a

    
    #kp = fe.extract(frame)
    #kp = fe.gridExtract(frame)
    kps, des = fe.gftt(frame)

    #kp = fe.extract(frame)
    for p in kps:
        #print(type(p))
        #round coords to nearest int and then draw
        #If using gftt feats only, use p[0]
        u,v = map(lambda x: int(round(x)), p.pt)
        print((u,v))
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
