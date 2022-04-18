import cv2
import numpy as np

#data/TransFill_Testing_Data/Real_Set/target/3_target.png
#data/TransFill_Testing_Data/Small_Set/target/21d5b2ebb33325f9_10_target_x1_GT.png

image_target = cv2.imread("data/TransFill_Testing_Data/Small_Set/target/22c25b89bb7005de_10_target_x1_GT.png", cv2.IMREAD_COLOR)
#data/TransFill_Testing_Data/Real_Set/hole/3_hole.png
#data/TransFill_Testing_Data/Small_Set/hole/hole.png
mask = cv2.imread("results/masked.png", cv2.IMREAD_GRAYSCALE) /255
mask = np.expand_dims(1-mask, -1).astype(np.uint8)
masked_target = image_target * mask
#data/TransFill_Testing_Data/Real_Set/source/3_source.png
#data/TransFill_Testing_Data/Small_Set/source/21d5b2ebb33325f9_10_target_x1_REFO.png
image_ref = cv2.imread("data/TransFill_Testing_Data/Small_Set/source/22c25b89bb7005de_10_target_x1_REFO.png", cv2.IMREAD_COLOR)

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

reference_pts = np.float32([ reference_kps[m.queryIdx].pt for m in matches ])
target_pts = np.float32([ target_kps[m.trainIdx].pt for m in matches ])
H, inliers= cv2.findHomography(reference_pts, target_pts, cv2.RANSAC,ransacReprojThreshold=5)
inliers = inliers.ravel().tolist()

out = cv2.drawMatches(image_ref,reference_kps,masked_target,target_kps,matches,None, matchesMask=inliers, 
    matchColor=(0,0,255),
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("results/ransac_matches.png", out)

warped = cv2.warpPerspective(image_ref, H, (masked_target.shape[1], masked_target.shape[0]))
cv2.imwrite("results/warp_perspective.png", warped)

result = np.zeros_like(warped)
mask = np.broadcast_to(mask, masked_target.shape)
result[mask==1] = masked_target[mask==1]
result[mask==0] = warped[mask==0]
cv2.imwrite("results/foreground.png", result)

# stitcher = cv2.Stitcher_create(cv2.STITCHER_SCANS)
# status, out = stitcher.stitch( [image_ref, masked_target], [np.ones_like(mask)*255, mask*255])
# print(status)

# cv2.imwrite("results/stitched.png", out)
# #Find Mask Corners
# def rough_mask(mask, shapes):
#     indices = np.argwhere(1-mask[:,:,0])
#     c1 = shapes[0]                  #Upper most
#     c2 = -1                         #Lower most
#     c3 = shapes[1]                  #Left most
#     c4 = -1                         #Right most
#     for index in indices:
#         if index[0] < c1:
#             c1 = index[0]
#         if index[0] > c2:
#             c2 = index[0]
#         if index[1] < c3:
#             c3 = index[1]
#         if index[1] > c4:
#             c4 = index[1]

#     r_mask = np.zeros_like(mask)
#     r_mask[c1:c2+1, c3:c4+1] = 1
#     center = (int((c3+c4)/2), int((c1+c2)/2)) 
#     return r_mask, center

# r_mask,center = rough_mask(mask, masked_target.shape)

# cv2.imwrite("temp.png", warped*r_mask)
# masked_target = masked_target + 200*(1-mask)
# out = cv2.seamlessClone(result, masked_target, (1-mask)*255, center, cv2.NORMAL_CLONE)  # r_mask*255 
# cv2.imwrite("results/poisson.png", out)

# img2 = cv2.circle(masked_target,center,20,tuple(np.random.randint(0,255,3).tolist()),-1)
# cv2.imwrite("results/img2.png", img2)

# pts1 = np.int32(reference_pts)
# pts2 = np.int32(target_pts)
# F, F_inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# pts1 = pts1[F_inliers.ravel()==1]
# pts2 = pts2[F_inliers.ravel()==1]



# h1, w1, _ = image_ref.shape
# h2, w2, _ = masked_target.shape
# _, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize=(w1+w2, h1+h2), threshold=0)

# ref_undistorted = cv2.warpPerspective(image_ref, H1, (w1+w2, h1+h2))
# target_undistorted = cv2.warpPerspective(masked_target, H2, (w2+w1, h2+h1))
# cv2.imwrite("results/undistorted_ref.png", ref_undistorted)
# cv2.imwrite("results/undistorted_target.png", target_undistorted)

# p1 = np.append(pts1[1],1)
# p2 = np.append(pts2[1],1)

# p1 = np.expand_dims(p1, 1)
# p2 = np.expand_dims(p2, 1)
# p1 = np.dot(H1, p1)
# p2 = np.dot(H2, p2)

# p1 = (p1 / p1[-1])[0:2]
# p2 = (p2 / p2[-1])[0:2]

# print()

# def drawlines(img1,img2,lines,pts1,pts2):
#     ''' img1 - image on which we draw the epilines for the points in img2
#         lines - corresponding epilines '''
#     r,c, _ = img1.shape
#     for r,pt1,pt2 in zip(lines,pts1,pts2):
#         color = tuple(np.random.randint(0,255,3).tolist())
#         x0,y0 = map(int, [0, -r[2]/r[1] ])
#         x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
#         img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
#         img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
#         img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
#     return img1,img2

# lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
# lines1 = lines1.reshape(-1,3)
# img5,img6 = drawlines(ref_undistorted,target_undistorted,lines1,pts1,pts2)
# # Find epilines corresponding to points in left image (first image) and
# # drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
# lines2 = lines2.reshape(-1,3)
# img3,img4 = drawlines(target_undistorted,ref_undistorted,lines2,pts2,pts1)

# cv2.imwrite("epip1.png", img5)
# cv2.imwrite("epip2.png", img3)
