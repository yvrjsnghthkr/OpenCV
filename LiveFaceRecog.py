import cv2 as cv
import numpy as np
import face_recognition
import os

path = '/Users/yvrjsnghthkr/Desktop/yvcodes/OpenCV/Advance/faces'
faces = []
known_faces = []
myList = os.listdir(path)
#print(myList)
#USe 'find . -name ".DS_Store" -delete' if .DS_Store file appears

for cls in myList:
    current_img = cv.imread(f'{path}/{cls}')
    faces.append(current_img)
    known_faces.append(os.path.splitext(cls)[0])
#print(known_faces)
def findEncodings(images):
    encodefaces = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodefaces.append(encode)
    return encodefaces

knownEncodings = findEncodings(faces)
#print(len(knownEncodings))

vid = cv.VideoCapture(0)
while True:
    ret, frame = vid.read()
    images = cv.resize(frame, (0, 0), None, 0.25, 0.25)
    images = cv.cvtColor(images, cv.COLOR_BGR2RGB)

    CurrentFrameFaces = face_recognition.face_locations(images)
    CurrentFrameEncode = face_recognition.face_encodings(images, CurrentFrameFaces)

    for FaceEncodes, FaceLoc in zip(CurrentFrameEncode, CurrentFrameFaces):
        matches = face_recognition.compare_faces(knownEncodings, FaceEncodes)
        faceDis = face_recognition.face_distance(knownEncodings, FaceEncodes)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = known_faces[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = FaceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 225, 0), 2)
            cv.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 0, 0),2)
            cv.putText(frame, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0,225), 1)

    cv.imshow('Webcam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
vid.release()
cv.destroyAllWindows()
