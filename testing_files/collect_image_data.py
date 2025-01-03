
import cv2
import pathlib

#face train data from python library
cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"


#must convert it to string
detector = cv2.CascadeClassifier(str(cascade_path))


camera = cv2.VideoCapture(0) # can put file names in here

while True:
    # ignore first argument  use second as camera data
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.07, minNeighbors=4, minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
        )
    for (x, y, width, height) in faces:
        cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()
