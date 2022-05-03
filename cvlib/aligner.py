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

        #Warping and Inpainting
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

        def rough_mask(mask, shapes):
            indices = np.argwhere(1-mask[:,:,0])
            c1 = shapes[0]                  #Upper most
            c2 = -1                         #Lower most
            c3 = shapes[1]                  #Left most
            c4 = -1                         #Right most
            for index in indices:
                if index[0] < c1:
                    c1 = index[0]
                if index[0] > c2:
                    c2 = index[0]
                if index[1] < c3:
                    c3 = index[1]
                if index[1] > c4:
                    c4 = index[1]

            r_mask = np.zeros_like(mask, dtype=np.uint8)
            c1 = c1 - 10
            c3 = c3 - 10
            c4 = c4 + 10
            r_mask[c1:c2+1, c3:c4+1] = 1
            center = (int((c3+c4)/2), int((c1+c2)/2)) 
            return r_mask, center, (c1,c2,c3,c4)

        r_mask,center,(c1,c2,c3,c4) = rough_mask(mask, self.target.shape)
        #r_mask = cv2.dilate((r_mask*255).copy(), None, iterations=100) / 255
        cv2.imwrite("results/cropped.png", self.target[c1:c2+1, c3:c4+1])
        cv2.imwrite("results/mask.png", (1-self.mask[c1:c2+1, c3:c4+1])*255)
        context_im = self.target[c1:c2+1, c3:c4+1].copy()
        mask_cropped = (1-self.mask)[c1:c2+1, c3:c4+1].copy()
        reference_cropped = warped[c1:c2+1, c3:c4+1].copy()
        conf_finder = ConfidenceFinder(context_im, mask_cropped, reference_cropped)

        # backg = self.target.copy()
        # backg[c1:c2+1, c3:c4+1] = back
        # cv2.imwrite("temp.png", backg)
        #masked_target = masked_target + 200*(1-mask)
        # out = cv2.seamlessClone(warped, backg, r_mask*255 ,center, cv2.NORMAL_CLONE)# r_mask*255,
        # cv2.imwrite("seamless.png", out)
        
        return aligned