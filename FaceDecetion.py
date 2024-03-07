from PIL import Image,ImageDraw
import face_recognition

image = face_recognition.load_image_file("/Users/yvrjsnghthkr/Downloads/IMG_3772.jpg")
face_locations = face_recognition.face_locations(image)

number_of_faces = len(face_locations)
print("I found {} faces in the picture.".format(number_of_faces))

pil_image=Image.fromarray(image)
draw=ImageDraw.Draw(pil_image)
for face_location in face_locations:
        top, right, bottom, left = face_location
        print("A face has been located at:")
        print(f"\tTop: {top}")
        print(f"\tLeft: {left}")
        print(f"\tBottom: {bottom}")
        print(f"\tRight: {right}")
        draw.rectangle(((left, top),(right, bottom)),outline=(0,225,0))
pil_image.show()        