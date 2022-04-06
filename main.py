from cvlib import face_detector, mask_generator
import cv2
import copy

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

kp1, des1 = sift.detectAndCompute(masked_target, mask)
out = cv2.drawKeypoints(masked_target, kp1, 0, (0, 255, 0),flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
cv2.imwrite("results/target_kps.png", out)

best_score= 0
detected_matches = []
detected_face = -1
detected_kps = -1
for face in faces:
    x,y,w,h = face
    face_img = image_ref[y:y+h, x:x+w]
    kp2, des2 = sift.detectAndCompute(face_img, None)
    matches = matcher.knnMatch(des1,des2,k=2)

    #Knn ratio test
    best_mathces = []
    for nearest, second_nearest in matches:
        if (nearest.distance / second_nearest.distance < 0.7):
            best_mathces.append([nearest])
    if ( len(best_mathces) > best_score):
        best_score = len (best_mathces)
        detected_matches = best_mathces
        detected_face = face_img
        detected_kp = kp2
    
out = cv2.drawMatchesKnn(masked_target,kp1,detected_face,detected_kp,detected_matches,None)
cv2.imwrite("results/matches.png", out)

