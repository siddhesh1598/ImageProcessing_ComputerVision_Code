# USAGE
# python knn.py --dataset ../Datasets/kaggle_dogs_vs_cats/

# import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import numpy as np
import argparse
import cv2
import os
import random

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
    help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
    help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
    help="# of jobs for kNN")
args = vars(ap.parse_args())

# describe the images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)
# load a batch of images
imagePaths = imagePaths[:1000]

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
    random_state=42)

# train kNN
print("[INFO] training kNN...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"],
    n_jobs=args["jobs"])
model.fit(trainX, trainY)

# print classification report
print(classification_report(testY, model.predict(testX),
    target_names=le.classes_))
