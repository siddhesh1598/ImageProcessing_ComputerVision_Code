# import
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import numpy as np
import argparse
import glob
import os

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--models", required=True,
	help="path to models directory")
args = vars(ap.parse_args())

# load dataset and scale it to [0, 1]
(testX, testY) = cifar10.load_data()[1]
testX = testX.astype("float") / 255.0

# initialize label names
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# convert labels to integers
lb = LabelBinarizer()
testY = lb.fit_transform(testY)

# initialize list of paths of models
modelPaths = os.path.sep.join(args["models"], "*.model")
modelPaths = list(glob.glob(modelPaths))
models = []

# loop over modelPaths and load the model, add it 
# to list of models
for (i, modelPath) in enumerate(modelPaths):
	print("[INFO] loading model {}/{}".format(i+1, len(modelPaths)))
	models.append(load_model(modelPath))

# evaluate models
predictions = []

for model in models:
	predictions.append(model.predict(testX, batch_size=64))

# average the probabilities across all models
predictions = np.mean(predictions, axis=0)
print(classification_report(testY.argmax(axis=0),
	predictions.argmax(axis=0), target_labels=labelNames)

