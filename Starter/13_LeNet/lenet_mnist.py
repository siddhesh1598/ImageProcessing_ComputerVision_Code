# import
from lenet import LeNet
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarzier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

# get the MNIST dataset
print("[INFO] loading dataset...")
dataset = datasets.fetch_mldata("MNIST Original")
data = dataset.data
labels = dataset.target.astype("int")

# normalize the data
data = data.astype("float") / 255.0


# split the dataset 
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25)

# convert target integers to vectors
le = LabelBinarzier()
train = le.fit_transform(trainY)
testY = le.transform(testY)

# compile model
print("[INFO] compiling model...")
opt = SGD(0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizers=opt,
	metrics=["accuracy"])

# train 
print("[INFO] training model...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=128, epochs=20, verbose=1)

# evaluate 
print("[INFO] evaluating model...")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

# plot training and loss accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Acc")
plt.legend()
plt.show()