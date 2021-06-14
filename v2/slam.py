#!/usr/bin/env python3

import cv2

orb = cv2.ORB_create()
print(dir(orb))

def process_frame(frame):
    cv2.imshow("image", frame)

    #detectAndCompute() -> list of keypoints, list of descriptors
    #https://docs.opencv.org/3.4/d0/d13/classcv_1_1Feature2D.html

    #keypoints:
    #https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html

    #kp.pt is the coordinates of the keypoint (type Point2f)
    #has tuple 
    #https://docs.opencv.org/3.4/dc/d84/group__core__basic.html#ga7d080aa40de011e4410bca63385ffe2a

    kp, des = orb.detectAndCompute(frame, None)
    for p in kp:
        #p is a tuple
        #round coords to nearest int and then draw
        u,v = map(lambda x: int(round(x)), p.pt)
        print((u,v))
        cv2.circle(frame, (u,v), radius=3, color=(0,255,0), thickness=-1)


if __name__ == "__main__" :
    print("test")
    fileName = 'fpv.avi'

    cap = cv2.VideoCapture(fileName)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
            cv2.waitKey(1)
        else:
            cap.release()
            break
