# USAGE
# python test_multiple_face.py --model face.model --image "path to image"

# import the necessary packages
import imutils
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import time

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
ap.add_argument("-m", "--model", required=True,	help="Path to trained model")
ap.add_argument("-s", "--scale", type=float, default=1.5, help="scale factor size")
args = vars(ap.parse_args())
 
# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (128,128)
model = load_model(args["model"])


def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image

	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)

		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break

		# yield the next image in the pyramid
		yield image


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# loop over the image pyramid
for resized in pyramid(image, scale=0.9):
	# loop over the sliding window for each layer of the pyramid
	for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
		# if the window does not meet our desired window size, ignore it
		if window.shape[0] != winH or window.shape[1] != winW:
			continue
		window = cv2.resize(window, (28, 28))
		window = window.astype("float") / 255.0
		window = img_to_array(window)
		window = np.expand_dims(window, axis=0)
		# classify the input image
		(notFace, face) = model.predict(window)[0]
		if face > 0.99:
			cv2.rectangle(image, (x, y), (x + winW, y + winH), (255, 255, 0), 2)

imgName = os.path.splitext(args["image"])[0] + "_output.jpg"
cv2.imwrite(imgName, image)
