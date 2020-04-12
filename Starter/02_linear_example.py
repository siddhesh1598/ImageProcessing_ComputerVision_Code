# Linear Classifier
# f(x, W, b) -> W.x + b
# x = (dimensions, 1)
# W = (classes, dimensions)
# b = (classes, 1)

# import 
import numpy as np
import cv2

# init labels
labels = ["dog", "cat"]
np.random.seed(1)

# init random weight and bias
W = np.random.randn(2, 3072)
b = np.random.randn(2)

# load image and flatten it
orig = cv2.imread("dog.jpg")
image = cv2.resize(orig, (32, 32)).flatten()

# calculate the score
scores = W.dot(image) + b

# print the score for each class
for (label, score) in zip(labels, scores):
	print("[INFO] {}: {:.2f}".format(label, score))

# draw the prediction on the image and show
cv2.putText(orig, "Label: {}".format(labels[np.argmax(scores)]),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
cv2.imshow("Image", orig)
cv2.waitKey(0)