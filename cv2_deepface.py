import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import  load_model
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace

# load model

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    result = DeepFace.analyze(img_path = frame , actions=["age", "gender", "emotion"], enforce_detection=False )

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_haar_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)

    emotion = result[0]
    # print(emotion)
    cv2.putText(frame,str(emotion["age"]),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.putText(frame,str(emotion["dominant_gender"]),(50,75),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.putText(frame,str(emotion["dominant_emotion"]),(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    # cv2.putText(frame,str(emotion["dominant_race"]),(50,125),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    cv2.imshow('frame',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()