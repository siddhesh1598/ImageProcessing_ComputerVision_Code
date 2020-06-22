# add resources path to sys
import sys
sys.path.append("../")

# save plots in background
import matplotlib
matplotlib.use("Agg")

# import 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from resources.nn.conv.minivggnet import MiniVGGNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matpotlib.pyplot as plt
import numpy as np
import argparse
import os

# construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
	help="path to output directory")
ap.add_argument("-m", "--models", required=True,
	help="path to output models directory")
ap.add_argument("-n", "--num-models", type=int, default=5,
	help="# of models to train")
args = vars(ap.parse_args())

# load the dataset and scale the data in the range [0, 1]
print("[INFO] loading dataset...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert labels to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# initialize label names
labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

# construct the object for data augmentation
aug = ImageDataGenerator(rotation_range=10, width_shift+range=0.1,
	height_shift_range=0.1, horizontal_flip=True, 
	fill_mode="nearest")

# loop over the number of models to train
for i in range(args["num_models"]):
	print("[INFO] training model {}/{}".format(i+1, 
		args["num_models"]))

	# initialize the optimizer and the model
	opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, 
		nesterov=True)
	model = MiniVGGNet.build(width=32, height=32, depth=32, 
		classes=10)
	model.compile(loss="categorical_crossentropy", optimizer=opt,
		metrics["accuracy"])

	# train
	H = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
		validation_data=(testX, testY), epochs=40,
		steps_per_epoch=len(trainX) // 64, verbose=1)

	# save the model
	p = [args["model"], "model_{}.model".format(i)]
	model.save(os.path.sep.join(p))

	# evaluate
	predictions = model.predict(testX, batch_size=64)
	report = classification_report(testY.argmax(axis=-1),
		predictions.argmax(axis=-1), target_labels=labelNames)

	# save classification report
	p = [args["output"], "model_{}.txt".format(i)]
	f = open(os.path.sep.join(p), "w")
	f.write(report)
	f.close()

	# plot the training loss and accuracy
	p = [args["output"], "model_{}.png".format(i)]
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(np.arange(0, 40), H.history["loss"],
	label="train_loss")
	plt.plot(np.arange(0, 40), H.history["val_loss"],
	label="val_loss")
	plt.plot(np.arange(0, 40), H.history["accuracy"],
	label="train_acc")
	plt.plot(np.arange(0, 40), H.history["val_accuracy"],
	label="val_acc")
	plt.title("Training Loss and Accuracy for model {}".format(i))
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend()
	plt.savefig(os.path.sep.join(p))
	plt.close()
