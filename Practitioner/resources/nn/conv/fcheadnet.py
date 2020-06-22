# import
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class FCHeadNet:
	@staticmethod

	def build(baseModel, classes, D):
		headModel = baseModel.output
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(D, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)

		# add a softmax layer
		headModel = Dense(classes, activation="softmax")(headModel)

		return headModel
