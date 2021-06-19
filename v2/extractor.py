import cv2
import numpy as np

class FeatureExtractor(object):

    #Contains different methods for extracting keypoints from an img

    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures = 1000)
        self.bf = cv2.BFMatcher()
        self.last = None

    def extract(self, img):

        #Feature Extraction
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance = 3)

        #Compute Keypoints
        if feats is not None:
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        else:
            kps = []

        #Compute Keypoint, Descriptor pairs from Keypoints
        kps, des = self.orb.compute(img, kps)

        #Matching
        matches = None
        if self.last is None:
            self.last = {'kps': kps, 'des': des}
            return None

        #Generate match pairs between keypoints on current and previous frames
        matches = self.bf.match(des, self.last['des'])
        res = zip([kps[m.queryIdx] for m in matches], [self.last["kps"][m.trainIdx] for m in matches])

        #Update last
        self.last = {'kps': kps, 'des': des}

        #Return matched points
        return res
