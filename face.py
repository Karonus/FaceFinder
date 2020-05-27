import cv2
import numpy as np
import os
import datetime
import lxml.html 
import requests
import bs4

faceCascade = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,1920) # Width
cap.set(4,1080) # Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        
        scaleFactor = 1.2,
        minNeighbors = 5,     
        minSize = (20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imwrite("data/peopls/peopl_" + str(datetime.datetime.now().hour) + "." + str(datetime.datetime.now().minute) + "." + str(datetime.datetime.now().second) + "." + str(datetime.datetime.now().microsecond) + ".jpg", gray[y:y+h,x:x+w])
        

    cv2.imshow('Detector',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
