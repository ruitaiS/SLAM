import sys
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


class FeatureExtractor(object):

    #Contains different methods for extracting keypoints from an img

    def __init__(self):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, img):

        #Feature Extraction
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance = 3)

        if feats is None:
            return None
        else:
            #Compute Keypoint, Descriptor Pairs
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
            kps, des = self.orb.compute(img, kps)

            if self.last is None:
                #If no last, then update w/ current and return w/o matching
                self.last = {'kps': kps, 'des': des}
                return None
            else:
                #Generate match pairs between keypoints on current and previous frames
                matches = self.bf.knnMatch(des, self.last['des'], k=2) #not sure what k does

                res = []

        
                #Ratio Test
                #Could we try a version w/ stdev of distance or st?
                for m,n in matches:
                    if m.distance < 0.75*n.distance:
                        res.append((kps[m.queryIdx].pt, self.last["kps"][m.trainIdx].pt))

                #Filter Matches
                if len(res)>8:
                    res = np.array(res)

                    #Normalize coords
                    #Denormalized later in slam.py
                    res[:, :, 0] -= img.shape[0]//2
                    res[:, :, 1] -= img.shape[1]//2

                    #TODO: Replace with better error handling
                    #RN it even catches keyboard interrupts
                    try:
                        model, inliers = ransac((res[:,0], res[:,1]),FundamentalMatrixTransform, min_samples=8, residual_threshold=1, max_trials=100)
                        res = res[inliers]
                    except:
                        print("Matching error:", sys.exc_info()[0])
                        #Should we update last or reset it to none in the case of error?
                        #self.last = {'kps': kps, 'des': des}
                        self.last = None
                        return None

                #Update last
                self.last = {'kps': kps, 'des': des}
                #Return matched points
                return res
