# import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

class ShallowNet:
	@staticmethod

	def build(width, height, depth, classes):
		model = Sequential()
		inputShape = (height, width, depth)

		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(Flatten())
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		return model	