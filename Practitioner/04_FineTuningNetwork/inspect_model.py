# add resources path to sys
import sys
sys.path.append("../")

# import
from tensorflow.keras.applications import VGG16
import argparse

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int,
	default=1, help="whether or not to include the top of CNN")
args = vars(ap.parse_args())

# load the model
print("[INFO] loading VGG16 model...")
model = VGG16(weights="imagenet",
	include_top=args["include_top"] > 0)

# loop over the layers in the network
print("[INFO] showing layers...")
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer.__class__.__name__))
