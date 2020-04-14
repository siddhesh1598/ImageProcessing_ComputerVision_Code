# import
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse

# define sigmoid activation function
def sigmoid_activation(x):
	return 1.0 / (1 + np.exp(-x))

# predict the output
def predict(X, W):
	# get y_hat by dot product
	pred = sigmoid_activation(X.dot(W))

	# apply step function to threshold the outputs 
	# to binary class label
	preds[preds <= 0.5] = 0
	preds[preds > 0.5] = 1

	return preds

# function to create mini-batches
def next_batch(X, y, batchSize):
	for i in np.arange(0, X.shape[0], batchSize):
		yield(X[i : i + batchSize], y[i : i + batchSize])

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=float, default=100,
	help="# of epochs")
ap.add_argument("-a", "--aplha", type=float, default=0.01,
	help="learning rate")
ap.add_argument("-b", "--batch-size", type=int, default=32,
	help="size of SGD mini-batches")
args = vars(ap.parse_args())

# generate dataset with: 
# 			1,000 data points
#			2 classes
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
	cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1)) # OR y = np.expand_dims(y, axis=1)

# insert a column of 1's as the last entry as to
# treat 'bias' as a trainable parameter with 'weights'
X = np.c_[X, np.ones(X.shape[0])]

# split the dataset into train/test
(trainX, testX, trainY, testY) = train_test_split(X, y,
	test_size=0.5, random_state=42)

# init weight matrix and list of losses
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)
losses = list()

# loop over epochs
for epoch in np.arange(0, args["epochs"]):
	# init a list to get total loss over an epoch
	epochLoss = list()

	# loop over data in mini-batches
	for (batchX, batchY) in next_batch(X, y, args["batch_size"]):
		# get preds by calculating dot product between
		# X and W and passing them through 
		# sigmoid activation function
		preds = sigmoid_activation(trainX.dot(W))

		# calculate error and loss
		error = preds - trainY
		epochLoss.append(np.sum(error ** 2))

		# calculate the gradient
		gradient = trainX.T.dot(error)

		# update the weight matrix
		W += -args["aplha"] * gradient

	# update loss value by taking avg loss over all batches
	loss = np.average(epochLoss)
	losses.append(loss)

	# show update
	if epoch == 0 or (epoch + 1) % 5 == 0:
		print("[INFO] epoch = {}, loss = {:.7f}".format(int(epoch+1),
			loss))

# evaluate the model
print("[INFO] evaluating model...")
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY, s=30)

# construct a figure for loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
