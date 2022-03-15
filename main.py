from cvlib.face_detector import face_detector
import cv2

image = cv2.imread("data/69996.png", cv2.IMREAD_COLOR)
faces = face_detector.detect_faces(image)

## Compare Target Face with Every Found Faces. Use SIFT or something else
