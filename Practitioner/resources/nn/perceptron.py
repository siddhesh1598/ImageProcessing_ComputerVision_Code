# import
import numpy as np

class Percentron:

	def __init__(self, N, alpha=0.1):
		# init the weight matrix
		self.W = np.random.randn(N + 1) / np.sqrt(N)
		self.alpha = alpha

	def step(self, x):
		# apply step function
		return 1 if x > 0 else 0

	def fit(self, X, y, epochs=10):
		# insert a column of 1's asthe last entry
		# for the X input for bias-trick
		X = np.c_[X, np.ones(X.shape[0])]

		# loop over each epoch
		for epoch in np.arange(epochs):
			# loop over individual data point
			for (x, target) in zip(X, y):
				p = self.step(np.dot(x, self.W))

				# check if prediction is wrong
				if p != target:
					# calculate error
					error = p - target
					# update weights
					self.W += -self.alpha * error * x

	def predict(self, X, addBias=True):
		# ensure the input is a matrix
		X = np.atleast_2d(X)

		# check is bias column should be added
		if addBias:
			X = np.c_[X, np.ones(X.shape[0])]

		return self.step(np.dot(X, self.W))