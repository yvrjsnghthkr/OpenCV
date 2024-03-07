import cv2 as cv
import numpy as np
import face_recognition

vid_capture=cv.VideoCapture(0)
yv_image = face_recognition.load_image_file("/Users/yvrjsnghthkr/Desktop/YuvrajRecog.jpeg")
yv_encoding= face_recognition.face_encodings(yv_image)[0]
known_face_encodings=[
    yv_encoding
]
known_face_names=[
    "yv"
]
while True:
    ret, frame=vid_capture.read()
    rgb_frame=frame[:,:,::-1]
    face_locations= face_recognition.face_locations(rgb_frame)
    face_encodings=face_recognition.face_encodings(rgb_frame,face_locations)
    
    for face_encoding in face_encodings:
        matches=face_recognition.compare_faces(known_face_encodings,known_face_names)
        name="Unknown"
        if matches:
            name=known_face_names[0]
    for (top,right ,bottom,left), name in zip(face_locations,face_encodings):
        if True in matches:
            first_match_index=matches.index(True)
            name=known_face_names[first_match_index]
            print("Match found:", name)
        cv.rectangle(frame,(left,bottom-35),(right,bottom),(0,0,225),cv.FILLED)
        font=cv.FONT_HERSHEY_COMPLEX
        cv.putText(frame,name,(left+6,bottom-6),font,1.0,(225,225,225),1)
    cv.imshow('Vid',frame)
    
    if cv.waitKey(1) == ord('q'):
        break
vid_capture.release()
cv.destroyAllWindows()