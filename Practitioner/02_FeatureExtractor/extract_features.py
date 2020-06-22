# add resources path to sys
import sys
sys.path.append("../")

# import
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from resources.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--output", required=True,
	help="path to output HDF5 file")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer-size", type=int, default=1000,
	help="size of feature extraction buffer")
args = vars(ap.parse_args())

# store the batch size
bs = args["batch_size"]

# grab the image paths
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

# extract the labels from the path and encode the labels
labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the VGG model
print("[INFO] loading VGG model...")
model = VGG16(weights="imagenet", include_top=False)

# init HDF5 dataset
dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7),
	args["output"], dataKey="features", 
	bufSize=args["buffer_size"])

# store the class label names
dataset.storeClassLabels(le.classes_)

# init the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
	widgets=widgets).start()

# loop over the images
for i in np.arange(len(imagePaths), bs):
	# extract the images and the labels in batches
	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	# loop over the images and labels in the data
	for (j, imagePath) in enumerate(batchPaths):
		# load and preprocess the image
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		# add the image to the batches
		batchImages.append(image)

	# pass the batch to extract output features
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)

	# flatten the output features from the MAXPOOL layer
	features = features.reshape((features.shape[0], 512 * 7 * 7))

	# add the features and the labels to the dataset
	dataset.add(features, batchLabels)
	pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()