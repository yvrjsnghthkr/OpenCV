from PIL import Image,ImageDraw,ImageFont
import face_recognition

yv_image = face_recognition.load_image_file("/Users/yvrjsnghthkr/Desktop/YuvrajRecog.jpeg")
yv_encodings= face_recognition.face_encodings(yv_image)[0]
known_face_encodings=[
    yv_encodings
]
known_face_names=[
    "yv"
]
image = face_recognition.load_image_file("/Users/yvrjsnghthkr/Desktop/Yuvraj.jpeg")
face_locations = face_recognition.face_locations(image)
face_encodings=face_recognition.face_encodings(image)

pil_image=Image.fromarray(image)
draw=ImageDraw.Draw(pil_image)

for (top,right,bottom,left), face_encoding in zip(face_locations,face_encodings):    
    matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
    name="Unknown"
    if True in matches:
        first_match_index=matches.index(True)
        name=known_face_names[first_match_index]
    draw.rectangle(((left, top),(right, bottom)),outline=(0,225,0))
    print(name)
    text_width,text_height=draw.textsize(name)
    draw.rectangle(((left, bottom -text_height-10),(right,bottom)),fill=(0,0,225),outline=(0,0,225))
    draw.text((left+6,bottom-text_height-5),name,fill=(225,225,225,225))
del draw
pil_image.show()