# import
import h5py
import os

class HDF5DatasetWriter:

	def __init__(self, dims, outputPath, dataKey="images",
		bufSize=1000):
		# if dataset exists, raise error
		if os.path.exists(outputPath):
			raise ValueError("The entered output path already \
				exists. Delete the file before continuing, outputPath")

		# open the hdf5 file and create two datasets
		# one to store image features
		# another to store class labels
		self.db = h5py.File(outputPath, "w")
		self.data = self.db.create_dataset("labels", dims,
			dtype="float")
		self.labels = self.db.create_dataset("labels", (dim[0],),
			dtype="int")

		# store the buffer size and init the buffer
		self.bufSize = bufSize
		self.buffer = {"data": [], "labels": []}
		self.idx = 0

	def add(self, rows, labels):
		# add rows and labels to the buffer
		self.buffer["data"].extend(rows)
		self.buffer["labels"].extend(labels)

		# check to see if the buffer needs to be flushed
		if len(self.buffer["data"]) >= self.bufSize:
			self.flush()

	def flush(self):
		# write the buffer to the disk and reset the buffer
		i = self.idx + len(self.buffer["data"])
		self.data[self.idx:i] = self.buffer["data"]
		self.labels[self.idx:i] = self.buffer["labels"]
		self.idx = i
		self.buffer = {"data": [], "labels": []}

	def storeClassLabels(self, classLabels):
		# create a dataset to store the actual class label names,
		# then store the class labels
		dt = h5py.special_dtype(vlen=unicode)
		labelSet = self.db.create_dataset("label_names",
			(len(classLabels),), dtype=dt)
		labelSet[:] = classLabels

		def storeClassLabels(self, classLabels):
			# create a dataset to store the actual class label names,
			# then store the class labels
			dt = h5py.special_dtype(vlen=unicode)
			labelSet = self.db.create_dataset("label_names",
			(len(classLabels),), dtype=dt)
			labelSet[:] = classLabels

		self.db.close()