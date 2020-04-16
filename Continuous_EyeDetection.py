# Angkan Biswas
# 16.04.2020
# To mark eye in taken picture.
# Note: 1. Download 'haarcascade_eye_tree_eyeglasses.xml'
# $ wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml
# 2. Install 'opencv-contrib-python'
# $ pip install opencv-contrib-python

import cv2

# Load pre-trained model
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml') 

camera = cv2.VideoCapture(0)
for i in range(200):
	_, picture = camera.read()							# Take a picture

	eyes = model.detectMultiScale(picture)

	for x, y, w, h  in eyes:
		cv2.rectangle(picture, (x, y), (x + w, y + h), (255, 0, 0), 2)

	cv2.imshow("MyEye", picture)	
	cv2.waitKey(1)									# Hold display window for a while

camera.release()
