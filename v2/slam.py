#!/usr/bin/env python3

import cv2


def process_frame(frame):
    #Do Stuff w/ frame data
    print(frame.shape)

if __name__ == "__main__" :
    print("test")
    fileName = 'fpv.avi'

    cap = cv2.VideoCapture(fileName)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
