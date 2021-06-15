#!/usr/bin/env python3

import cv2


class FeatureExtractor(object):

    #Contains different methods for extracting keypoints from an img

    def __init__(self):
        self.orb = cv2.ORB_create(1000)

        #Used for gridExtract
        self.GX = 16 # grid width
        self.GY = 16 # grid height


    #Subdivides image into grid and extracts keypoints from each
    def gridExtract(self, img):
        rkp = [] #returned keypoints
        for ry in range(0, img.shape[0], img.shape[0]//self.GY):
            for rx in range(0, img.shape[1], img.shape[1]//self.GX):
                gridSection = img[ry:ry+self.GY, rx:rx+self.GX],
                #print(gridSection)
                kp = self.orb.detect(gridSection, None)
                #print(type(kp))
                for p in kp:
                    print(type(p))
                    #offset by grid section origin coords
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    rkp.append(p)
        return rkp

fe = FeatureExtractor()

def process_frame(frame):

    #detectAndCompute() -> list of keypoints, list of descriptors
    #https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

    #keypoints:
    #https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

    #kp.pt is the coordinates of the keypoint (type Point2f)
    #has tuple 
    #https://docs.opencv.org/3.4/dc/d84/group__core__basic.html#ga7d080aa40de011e4410bca63385ffe2a

    kp = fe.gridExtract(frame)

    for p in kp:
        #p is a tuple
        #round coords to nearest int and then draw
        u,v = map(lambda x: int(round(x)), p.pt)
        print((u,v))
        cv2.circle(frame, (u,v), radius=3, color=(0,255,0), thickness=-1)


if __name__ == "__main__" :
    #VideoCapture(fileName) for use on pre-recorded video
    #VideoCapture(0) pulls from webcam
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
