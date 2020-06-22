# add resources path to sys
import sys
sys.path.append("../")

# import
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from resources.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor
from resources.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
from resources.datasets.simpledatasetloader import SimpleDatasetLoader
from resources.nn.conv.fcheadnet import FCHeadNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to inout dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
args = vars(ap.parse_args())

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# grab the list of images and extract their class labels
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# init the image processor
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset and normalize it
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, 
	labels, test_size=0.25, random_state=42)
	
# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# load the VGG16 base model
baseModel = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# init new head model
headModel = FCHeadNet(baseModel, len(classNames), 256)

# combine the model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all the layers in the baseModel and
# freese the weights
for layer in baseModel.layers:
	layer.trainable = False

# compile the model
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy",
	optimizer=opt, metrics=["accuracy"])

# train the model
print("[INFO] training model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), epochs=25,
	steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=0),
	predictions.argmax(axis=0),
	target_names=classNames))

# now unfreese the last set of CONV layers 
for layer in baseModel.layers[15:]:
	layer.trainable = True

# compile the model again
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy",
	optimizer=opt, metrics=["accuracy"])

# train the model again
print("[INFO] fine-tuning the model...")
model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY), epochs=100,
	steps_per_epoch=len(trainX) // 32, v

# evaluate the model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), 
	target_names=classNames))

# save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])

