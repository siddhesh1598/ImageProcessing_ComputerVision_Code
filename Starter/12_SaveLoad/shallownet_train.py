# import 
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array
from shallownet import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import random

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
help="path to output model")
args = vars(ap.parse_args())

# grab the list of images that we’ll be describing
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
    image = img_to_array(image)
    label = imagePath.split(os.path.sep)[-1].split(".")[0]

    data.append(image)
    labels.append(label)

    if i != 0 and i % 1000 == 0:
        print("[INFO] processing {}/{}...".format(i, len(imagePaths)))
        # print(labels)

data = np.array(data)
data = data.astype("float") / 255.0
labels = np.array(labels)

# split the dataset into train/test
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25)

# convert the target int to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# compile model
print("[INFO] compiling model...")
opt = SGD(0.005)
model = ShallowNet.build(width=32, height=32, depth=3, 
	classes=2)
model.compile(loss=["sparse_categorical_crossentropy"], optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=32, epochs=100, verbose=1)

# save the model
print("[INFO] saving model...")
model.save(args["model"])

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), 
	target_names=["cat", "dog"]))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()