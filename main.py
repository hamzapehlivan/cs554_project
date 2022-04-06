from cvlib import face_detector, mask_generator
import cv2
import numpy as np

image_target = cv2.imread("data/target.png", cv2.IMREAD_COLOR)
mask = mask_generator.RandomMask(image_target.shape[0])
masked_target = image_target * mask
cv2.imwrite("results/masked_image.png", masked_target)
image_ref = cv2.imread("data/reference.png", cv2.IMREAD_COLOR)

## Find Faces
faces = face_detector.detect_faces(image_ref)
face_detector.draw_faces(image_ref, faces)


## Compare Target Face with Every Found Faces with SIFT
sift = cv2.SIFT_create()
matcher = cv2.BFMatcher.create(normType = cv2.NORM_L2, crossCheck=False)

target_kps, target_desc = sift.detectAndCompute(masked_target, mask)
out = cv2.drawKeypoints(masked_target, target_kps, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite("results/target_kps.png", out)

best_score= 0
matches = []
reference_face = -1
reference_kps = -1
for face in faces:
    x,y,w,h = face
    face_img = image_ref[y:y+h, x:x+w]
    kps, desc = sift.detectAndCompute(face_img, None)
    knn_matches = matcher.knnMatch(desc,target_desc,k=2)

    #Knn ratio test
    best_mathces = []
    for nearest, second_nearest in knn_matches:
        if (nearest.distance / second_nearest.distance < 0.7):
            best_mathces.append(nearest)
    if ( len(best_mathces) > best_score):
        best_score = len (best_mathces)
        matches = best_mathces
        reference_face = face_img
        reference_kps = kps
    
out = cv2.drawMatches(reference_face,reference_kps,masked_target,target_kps,matches,None)
cv2.imwrite("results/matches.png", out)

reference_pts = np.float32([ reference_kps[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
target_pts = np.float32([ target_kps[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
H, inliers= cv2.findHomography(reference_pts, target_pts, cv2.RANSAC,ransacReprojThreshold=3)
inliers = inliers.ravel().tolist()

out = cv2.drawMatches(reference_face,reference_kps,masked_target,target_kps,matches,None, matchesMask=inliers)
cv2.imwrite("results/ransac_matches.png", out)

warped = cv2.warpPerspective(reference_face, H, (masked_target.shape[0], masked_target.shape[1]))
cv2.imwrite("results/warp_perspective.png", warped)

result = np.zeros_like(masked_target)
mask = np.broadcast_to(mask, masked_target.shape)
result[mask==1] = masked_target[mask==1]
result[mask==0] = warped[mask==0]

cv2.imwrite("results/result.png", result)

print()
