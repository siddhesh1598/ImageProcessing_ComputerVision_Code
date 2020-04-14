# import
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imutils import paths
import cv2
import os
import numpy as np
import argparse
import random

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to the dataset")
args = vars(ap.parse_args())

# grab the list of input paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
# load a batch of images
imagePaths = imagePaths[:100]

# preprocess the images
data = []
labels = []

for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    data.append(image)
    labels.append(label)

    if i != 0 and i % 1000 == 0:
        print("[INFO] processing {}/{}...".format(i, len(imagePaths)))
        # print(labels)

data = np.array(data)
data = data.reshape((data.shape[0], 3072))
labels = np.array(labels)

# encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)

# split the dataset into train/test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25,
	random_state=5)

# lopp over set of regularizers
for r in (None, "l1", "l2"):
	# train SGD Classifier with softmax loss function and the 
	# specified regularization function
	print("[INFO] training model with '{}' penalty".format(r))
	model = SGDClassifier(loss="log", penalty=r, max_iter=10,
		learning_rate="constant", eta0=0.01, random_state=42)
	model.fit(trainX, trainY)

	# evaluate the classifier
	print("[INFO] evaluating the model...")
	acc = model.score(testX, testY)
	print("[INFO] '{}' prnalty accuracy: {:.2f}".format(r, acc*100))