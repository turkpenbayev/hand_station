import cv2
import numpy as np
cam = cv2.VideoCapture(0)

x = 60
y = 0
x2, y2 = x+300, y+400

id = 4

counter = 0
while(True):
    ret, img = cam.read()

    cv2.rectangle(img, (x, y), (x2, y2), (0,255,0),3)


    hand = img[y:y2, x:x2]
    gray = cv2.cvtColor(hand, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (24, 32))
    

    cv2.imshow('frame',img)
    cv2.imshow('hand', gray)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    if counter % 5 == 0:
        cv2.imwrite("data/{}-{}.jpg".format(id, counter//10), gray)

    if counter // 5 > 500:
        break

    counter+=1
