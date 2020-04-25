#	Angkan Biswas
#	25.04.2020
#	--------------------------------------------
#	Guess about the class (objects' name) of 
#	an image using pre-trained model VGG16.
#	-------------------------------------------

'''	Load necessary modules.	'''
from tensorflow.keras.applications.vgg16 import VGG16, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
import cv2
import numpy as np
import os

'''	Load the pre-trained model, VGG16.	'''
model = VGG16()
model.summary()

'''	Prapare a list of images in a folder	''' 
dirPath = '/home/dell/downloads/cat/'
imgList = os.listdir(dirPath)
n = len(imgList)
print(n)
for i in range(n):
	'''	Load image.	'''
	imgPath = dirPath + imgList[i]
	img = load_img(imgPath, target_size = (224, 224))

	'''	Turn loaded image into a numpy array.	'''
	rgbImg = np.array(img, dtype = np.uint8)

	'''	Turn 3D image matrix into 4D matrix so that it can be accepted by VGG16.'''
	img = np.reshape(rgbImg, (1, 224, 224, 3))	

	'''	Guess about the object's name which occupies the image most.'''
	guess = model.predict(img)
	classLabel = decode_predictions(guess)
	print(classLabel)
	classLabel = classLabel[0][0][1]

	'''	TensorFlow loads image into RGB format while OpenCV looks image in
	BGR format. So, we need to turn RGB image into BGR image, so that
	we can use cv2.imshow().
	'''
	bgrImg = cv2.cvtColor(rgbImg, cv2.COLOR_RGB2BGR)	

	'''	Display the image using the OpenCV.	'''
	cv2.imshow(classLabel, bgrImg)
	cv2.waitKey(0)
