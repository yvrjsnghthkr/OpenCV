import cv2 as cv
import numpy as np
import face_recognition

vid_capture=cv.VideoCapture(0)

while True:
    ret, frame=vid_capture.read()
    rgb_frame=frame[:,:,::-1]
    face_locations= face_recognition.face_locations(rgb_frame)
    #face_encodings=face_recognition.face_encodings(rgb_frame,face_locations,)

    for (top,right ,bottom,left), face_encoding in zip(face_locations,'''face_encodings'''):
        cv.rectangle(frame,(left,top),(right,bottom),(0,0,225),2)
    cv.imshow('Vid',frame)
    
    if cv.waitKey(1) & 0xFF==ord('q'):
        break
    
vid_capture.release()
cv.destroyAllWindows()