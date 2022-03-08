import cv2

def detect_faces(in_image):
    path = 'models/haarcascade_frontalface_alt.xml'
    haar = cv2.CascadeClassifier()
    haar.load(cv2.samples.findFile(path))
    image = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    faces = haar.detectMultiScale(image)
    return faces

#Test the function
if __name__ == '__main__':
    image = cv2.imread("data/69996.png", cv2.IMREAD_COLOR)
    faces = detect_faces(image)
    for face in faces:
        x,y,w,h = face
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 3)
    cv2.imwrite("results/detected_faces.png", image)