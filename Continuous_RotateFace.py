#	Angkan Biswas
#	20.04.2020
#	Rotated Face continuosly

import cv2
import imutils

model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load pre-trained model

camera = cv2.VideoCapture(0)							# Open a cemera.

degree = 0
for i in range(1000):
	_, frame = camera.read()						# Take a frame.
	resizeFrame = cv2.resize(frame,(512,512))				# Resize the frame.

	faces = model.detectMultiScale(resizeFrame)	
	for x, y, w, h  in faces:
		croppedFace = resizeFrame[x:x+w, y:y+h]

		rotateImg = imutils.rotate (croppedFace, degree)		# To rotate the frame		
	
		cv2.imshow('My Face' ,rotateImg)				# Display the frame.
		cv2.waitKey(1)							# Hold the display window for 1 millisecond
		degree = degree + 2
	
camera.release()								# Release the captured camera










