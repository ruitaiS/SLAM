import cv2
import numpy as np

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
                

                #res = zip([kps[m.queryIdx] for m in matches], [self.last["kps"][m.trainIdx] for m in matches])

                #Update last
                self.last = {'kps': kps, 'des': des}

                #Return matched points
                return res
