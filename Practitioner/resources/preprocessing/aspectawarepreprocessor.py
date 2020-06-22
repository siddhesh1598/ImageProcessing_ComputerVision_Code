# import
import imutils
import cv2

class AspectAwarePreprocessor:

	def __init__(self, width, height, inter=cv2.INTER_AREA):
		self.width = width
		self.heigh = height
		self.inter = inter

	def preprocess(self, image):
		(h, w) = image.shape[:2]
		dW, dH = 0, 0

		# if the width is smaller than the height, 
		# then resize along the width and then
		# update the deltas
		if w < h:
			image = imutils.resize(image, width=self.width,
				inter=self.inter)
			dH = int((image.shape[0] - self.height) / 2.0)

		# otherwise, the height is smaller than the width so
		# resize along the height and then update the deltas
		# to crop along the width
		else:
			image = imutils.resize(image, height=self.height,
				inter=self.inter)
			dW = int((image.shape[1] - self.width) / 2.0)

		# regrab the height and width after resizing 
		# to crop the image
		(h, w) = image.shape[:2]
		image = image[dH:h-dH, dW:w-dW]

		# finally, resize the image to the provided spatial
		# dimensions to ensure our output image is always 
		# a fixed size
		return cv2.resize(image, (self.width, self.height),
			interpolation=self.inter)