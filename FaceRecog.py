# -*- coding: utf-8 -*-
"""
Created on Mon May 21 07:58:19 2018

@author: ma
"""
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("haarcascade_smileface.xml")


def detect(gray,frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        detected_face = frame[y:y+h,x:x+w]
        detected_face_gray = gray[y:y+h,x:x+w] 
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        eyes = eye_cascade.detectMultiScale(detected_face_gray, 1.5, 5)
        for (x, y, w, h) in eyes:
            cv2.rectangle(detected_face,(x,y),(x+w,y+h),(0,255,0),2)
        smiles = smile_cascade.detectMultiScale(detected_face_gray, 1.3, 10)
        for (x, y, w, h) in smiles:
            cv2.rectangle(detected_face,(x,y),(x+w,y+h),(0,0,255),2)
    return frame

video_capture = cv2.VideoCapture(0)


while True:
    _, frame = video_capture.read()
    gray_image = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas = detect(gray_image,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'): # If we type on the keyboard:
        break # We stop the loop.

video_capture.release() # We turn the webcam off.
cv2.destroyAllWindows()
    
    





