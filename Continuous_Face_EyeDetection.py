# Angkan Biswas
# 16.04.2020
# To mark eye in taken picture.

# Note: 
#	1. Download 'haarcascade_frontalface_default.xml' & 'haarcascade_eye_tree_eyeglasses.xml'
# $ wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# $ wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml
#	2. Install 'opencv-contrib-python'
# $ pip install opencv-contrib-python

import cv2

# Load pre-trained model
faceModel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeModel = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml') 

camera = cv2.VideoCapture(0)
for i in range(200):
	_, picture = camera.read()							# Take a picture

	faces = faceModel.detectMultiScale(picture)
	eyes = eyeModel.detectMultiScale(picture)

	for x, y, w, h  in faces:
		cv2.rectangle(picture, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(picture, 'Face', (x, y-5), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.9, color = (0, 255, 0), thickness = 2)

	for x, y, w, h  in eyes:
		cv2.rectangle(picture, (x, y), (x + w, y + h), (255, 0, 0), 2)
		cv2.putText(picture, 'Eye', (x, y-5), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.9, color = (255, 0, 0), thickness = 2)

	cv2.imshow("Eye+Face", picture)	
	cv2.waitKey(1)									# Hold display window for a while

camera.release()
