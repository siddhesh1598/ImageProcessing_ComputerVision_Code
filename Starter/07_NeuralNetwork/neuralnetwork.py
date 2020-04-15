# import
import numpy as np

class NeuralNetwork:

	def __init__(self, layers, alpha=0.1):
		# init weights then store the network architecture
		# and learning rate
		self.W = []
		self.layers = layers
		self.alpha = alpha

		# loop over the layers leaving the last two layers
		for i in np.arange(len(layers) - 2):
			# randomly initialize a weight matrix 
			# connecting the number of nodes in each 
			# respective layer together, adding an 
			# extra node for the bias
			w = np.random.randn(layers[i] + 1, 
				layers[i + 1] + 1)
			self.W.append(w / np.sqrt(layers[i]))

		# the last 2 layers are special case where
		# the input layer needs a bias term but 
		# the output layer does not
		w = np.random.randn(layers[-2] + 1, layers[-1])
		self.W.append(w / np.sqrt(layers[-2]))

	def __repr__(self):
		# returns the network architecture as a string
		return "NeuralNetwork: {}".format(
			"-".join(str(l) for l in self.layers))

	def sigmoid(self, x):
		return 1.0 / (1 + np.exp(-x))

	def sigmoid_deriv(self, x):
		return x * (1 - x)

	def fit(self, X, y, epochs=100, displayUpdate=100):
		# insert a column of 1's as last entry in X
		X = np.c_[X, np.ones((X.shape[0]))]

		# loop over each epoch
		for epoch in np.arange(epochs):
			# get the data points
			for (x, target) in zip(X, y):
				self.fit_partial(x, target)

			# display training update
			if epoch == 0 or (epoch+1) % displayUpdate == 0:
				loss = self.calculate_loss(X, y)
				print("[INFO] epoch: {}, loss: {:.7f}".format(
					epoch + 1, loss))

	def fit_partial(self, x, y):
		# construct a list of activations for each layer
		# the first layer is just the feature vector
		A = [np.atleast_2d(x)]

		# FEEDFORWARD:
		# loop over the network
		for layer in np.arange(len(self.W)):
			# calculate the net input of the layer
			net = A[layer].dot(self.W[layer])

			# calculate the output by applying 
			# activation function
			out = self.sigmoid(net)

			# store the output
			A.append(out)

		# BACKPROPAGATION:
		# calculate the difference between our
		# prediction and the actual output
		error = A[-1] - y

		# construct a list of deltas
		# the first entry is the error of out output times
		# the derivation of the activation function of
		# that layer
		D = [error * self.sigmoid_deriv(A[-1])]

		# loop over the layers in reverse order
		# leaving the last 2
		for layer in np.arange(len(A) - 2, 0, -1):
			# delta of layer(t): 
			#	delta(t-1) 'dot' weights(t) * actfunderiv(t)
			delta = D[-1].dot(self.W[layer].T)
			delta = delta * self.sigmoid_deriv(A[layer])
			D.append(delta)
		# since we looped in reverse order, 
		# reverse the deltas
		D = D[::-1]

		# WEIGHT UPDATE:
		# loop over the layers
		for layer in np.arange(len(self.W)):
			# update the weights by taking dot product
			# of the layer activation with their delta
			# and multiplying with learning rate
			self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

	def predict(self, X, addBias=True):
		# init the output prediction as the input features
		# this we will then feedforward through the network
		p = np.atleast_2d(X)

		# check to see if bias should be added
		if addBias:
			p = np.c_[p, np.ones((p.shape[0]))]

		# loop over the layers in the network
		for layer in np.arange(len(self.W)):
			# conpute the net input and pass through
			# the activation function
			p = self.sigmoid(p.dot(self.W[layer]))

		return p	

	def calculate_loss(self, X, targets):
		# make predictions for input and calculate error
		targets = np.atleast_2d(targets)
		predictions = self.predict(X, addBias=False)
		loss = 0.5 * np.sum((predictions - targets) ** 2)

		return loss
