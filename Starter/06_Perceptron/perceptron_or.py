# inport
from perceptron import Percentron
import numpy as np

# create OR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [1]])

# train perceptron
print("[INFO] training perceptron...")
p = Percentron(X.shape[1], alpha=0.1)
p.fit(X, y, epochs=20)

# test perceptron
print("[INFO] testing perceptron...")
for (x, target) in zip(X, y):
	pred = p.predict(x)
	print("[INFO] data: {}, ground-truct: {}, \
		pred: {}".format(x, target[0], pred))