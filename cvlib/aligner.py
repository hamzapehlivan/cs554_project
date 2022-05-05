import cv2
import numpy as np
from cvlib.confidence_finder import ConfidenceFinder

class Aligner():
    def __init__(self, target, reference, mask, debug=True) -> None:
        self.target = target.copy()
        self.reference = reference.copy()
        self.mask = mask.copy()
        self.debug = debug
    
    def align(self, nndr_thres=0.75):
        #Create SIFT Features and Descriptors
        sift = cv2.SIFT_create()
        matcher = cv2.BFMatcher.create(normType = cv2.NORM_L2, crossCheck=False)
        target_kps, target_desc = sift.detectAndCompute(self.target, self.mask)
        reference_kps, reference_desc = sift.detectAndCompute(self.reference, None)

        #Get Best Matching Descriptors with Ratio Test
        knn_matches = matcher.knnMatch(reference_desc,target_desc,k=2)
        matches = []
        for nearest, second_nearest in knn_matches:
            if (nearest.distance / second_nearest.distance < nndr_thres):
                matches.append(nearest)
        
        #Homography Estimation with RANSAC
        reference_pts = np.float32([ reference_kps[m.queryIdx].pt for m in matches ])
        target_pts = np.float32([ target_kps[m.trainIdx].pt for m in matches ])
        H, inliers= cv2.findHomography(reference_pts, target_pts, cv2.RANSAC,ransacReprojThreshold=5)
        inliers = inliers.ravel().tolist()

        #Warping
        warped = cv2.warpPerspective(self.reference, H, (self.target.shape[1], self.target.shape[0]))
        aligned = np.zeros_like(warped)
        mask = np.broadcast_to(self.mask, self.target.shape)
        aligned[mask==1] = self.target[mask==1]
        aligned[mask==0] = warped[mask==0]
        
        if self.debug:
            out = cv2.drawKeypoints(self.target, target_kps, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
            cv2.imwrite("results/target_kps.png", out)
            out = cv2.drawMatches(self.reference,reference_kps,self.target,target_kps,matches,None)
            cv2.imwrite("results/matches.png", out)
            out = cv2.drawMatches(self.reference,reference_kps,self.target,target_kps,matches,None, matchesMask=inliers, 
                        matchColor=(0,0,255), flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite("results/ransac_matches.png", out)
            cv2.imwrite("results/warped.png", warped)
            cv2.imwrite("results/aligned.png", aligned)
            
        return aligned, warped