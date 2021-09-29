import sys
import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

def addOnes(x):
    #[[x,y]] -> [[x,y,1]]
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)



class FeatureExtractor(object):

    #Contains different methods for extracting keypoints from an img

    def __init__(self, frame, F):

        #From initial frame, calculate intrinsic parameters matrix & it's inverse
        self.K = np.array([[F, 0, frame.shape[0]//2],[0,F,frame.shape[1]//2],[0,0,1]])
        self.Kinv = np.linalg.inv(self.K)

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        #print(ret)
        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):

        #Feature Extraction
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis = 2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance = 3)

        if feats is None:
            return None
        else:
            #Compute Keypoint, Descriptor Pairs
            kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
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
                res = np.array(res)

                #Filter Matches
                #NOTE: AddOnes won't work on empty array
                #On init, res[:0,:] is empty; I *think* because it hasn't been populated with matched kp's yet
                #RN just check that it's non-empty as a workaround
                if len(res)>0 and len(res[:, 0,:])>0 and len(res[:, 1,:])>0:
                    #Normalize coords
                    #Denormalized later in slam.py
                    res[:,0,:] = np.dot(self.Kinv, addOnes(res[:, 0,:]).T).T[:,0:2]
                    res[:,1,:] = np.dot(self.Kinv, addOnes(res[:, 1,:]).T).T[:,0:2]


                    #TODO: Replace with better error handling
                    #RN it even catches keyboard interrupts
                    try:
                        model, inliers = ransac((res[:,0], res[:,1]),FundamentalMatrixTransform, min_samples=8, residual_threshold=1, max_trials=100)
                        res = res[inliers]
            
                        #TODO: Figure out wtf this is doing
                        s,v,d = np.linalg.svd(model.params)
                        print(v)
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
