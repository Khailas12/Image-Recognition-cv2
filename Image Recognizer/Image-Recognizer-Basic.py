import cv2
from random import randrange
trained_face_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
#img = cv2.imread("bill-gates.png")
img = cv2.imread("Happy-group-of-people.png")
img = cv2.resize(img, (700, 500))

greyscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(greyscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256),randrange(256), randrange(256)), 8)
#                                              (0, 255, 0), 8)   instead of this for getting random colors

print(face_coordinates)

cv2.imshow("Face Detector", img)   
cv2.waitKey()


print("Done")


