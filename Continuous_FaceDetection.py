# Angkan Biswas
# 16.04.2020
# To mark face in taken picture.
# Note: 1. Download 'haarcascade_frontalface_default.xml'
# $ wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
# 2. Install 'opencv-contrib-python'
# $ pip install opencv-contrib-python

import cv2

# Load pre-trained model
model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

camera = cv2.VideoCapture(0)
for i in range(100):
	_, picture = camera.read()					#	Take a picture

	faces = model.detectMultiScale(picture)

	for x, y, w, h  in faces:
		cv2.rectangle(picture, (x, y), (x + w, y + h), (255, 0, 0), 2)

	cv2.imshow("MyFace", picture)	
	cv2.waitKey(1)							#	Hold display window for a while

camera.release()
