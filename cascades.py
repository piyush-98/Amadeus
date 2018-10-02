import numpy as np
import cv2
import serial
import time
import struct
import pyrebase
config = {
  "apiKey": "apiKey",
  "authDomain": "bageera2018.firebaseapp.com",
  "databaseURL": "https://bageera2018.firebaseio.com",
  "storageBucket": "bageera2018.appspot.com",
  #"serviceAccount": "path/to/serviceAccountCredentials.json"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

palm_cascade = cv2.CascadeClassifier(r'C:\Users\PIYUSH\Desktop\palm.xml')
ArduinoSerial = serial.Serial('COM5',9600)
time.sleep(2)
#img = cv2.imread(r'C:\Users\PIYUSH\Desktop\h1.jpg')
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    print(frame.shape)
    cv2.imshow("frame",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    palmy = palm_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in palmy:
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #print(img)
        cv2.imshow('frame',frame)
        angle= ((x+w/2)/3.5)
        angle = int(angle)
        ArduinoSerial.write(struct.pack("B", angle))
        #db.child("angle").set({1:angle})
        #print(angle)
        #time.sleep(1)
    if cv2.waitKey(1) & 0xFF == ord('h'):
        break
   
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

