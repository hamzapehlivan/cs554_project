from cvlib import face_detector, mask_generator
import cv2
import numpy as np

image_target = cv2.imread("data/TransFill_Testing_Data/Real_Set/target/1_target.png", cv2.IMREAD_COLOR)
mask = cv2.imread("data/TransFill_Testing_Data/Real_Set/hole/1_hole.png", cv2.IMREAD_GRAYSCALE) /255
mask = np.expand_dims(1-mask, -1).astype(np.uint8)
masked_target = image_target * mask
image_ref = cv2.imread("data/TransFill_Testing_Data/Real_Set/source/1_source.png", cv2.IMREAD_COLOR)

cv2.imwrite("results/ref.png", image_ref)
cv2.imwrite("results/target.png", image_target)
cv2.imwrite("results/masked_image.png", masked_target)



## Compare Target Face with Every Found Faces with SIFT
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher.create(normType = cv2.NORM_L2, crossCheck=False)

target_kps, target_desc = sift.detectAndCompute(masked_target, mask)
out = cv2.drawKeypoints(masked_target, target_kps, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite("results/target_kps.png", out)


reference_kps, reference_desc = sift.detectAndCompute(image_ref, None)
knn_matches = matcher.knnMatch(reference_desc,target_desc,k=2)

#Knn ratio test
matches = []
for nearest, second_nearest in knn_matches:
    if (nearest.distance / second_nearest.distance < 0.75):
        matches.append(nearest)

out = cv2.drawMatches(image_ref,reference_kps,masked_target,target_kps,matches,None)
cv2.imwrite("results/matches.png", out)

reference_pts = np.float32([ reference_kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
target_pts = np.float32([ target_kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
H, inliers= cv2.findHomography(reference_pts, target_pts, cv2.RANSAC,ransacReprojThreshold=5)
inliers = inliers.ravel().tolist()

out = cv2.drawMatches(image_ref,reference_kps,masked_target,target_kps,matches,None, matchesMask=inliers)
cv2.imwrite("results/ransac_matches.png", out)

warped = cv2.warpPerspective(image_ref, H, (masked_target.shape[1], masked_target.shape[0]))
cv2.imwrite("results/warp_perspective.png", warped)

result = np.zeros_like(warped)
mask = np.broadcast_to(mask, masked_target.shape)
result[mask==1] = masked_target[mask==1]
result[mask==0] = warped[mask==0]
cv2.imwrite("results/result.png", result)

