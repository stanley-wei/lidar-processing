import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import torch

import config
from dataset import SegmentationDataset

imagePaths = sorted(os.path.join(config.IMAGE_DATASET_PATH,file) for file in os.listdir(config.IMAGE_DATASET_PATH))
maskPaths = sorted(os.path.join(config.MASK_DATASET_PATH,file) for file in os.listdir(config.MASK_DATASET_PATH))

dataset = SegmentationDataset(imagePaths=imagePaths, maskPaths=maskPaths,
    transforms=None, data_type="image")
num_classes = len(dataset.classes)
print(dataset.classes)

def squeeze(array):
	while array.shape[0]==1:
		array = np.squeeze(array, axis=0)
	return array	

def collapse_mask(mask):
	mask = squeeze(mask)
	composite_mask = np.zeros(mask.shape[1:3], dtype=np.float32)

	for i in range(mask.shape[1]):
		for j in range(mask.shape[2]):
			max_val = 0
			max_channel = 0
			for k in range(1,mask.shape[0]):
				if mask[k][i][j] >= max_val:
					max_val = mask[k][i][j]
					max_channel = k
			composite_mask[i][j] = max_channel

	print(np.unique(composite_mask))

	mask = np.zeros(composite_mask.shape, dtype=np.float32)
	for index, label in enumerate(dataset.classes):
		mask[composite_mask == label] = index
	mask *= 254 / len(dataset.classes)
	mask = np.asarray(mask, dtype=np.uint8)

	return composite_mask

def prepare_plot(origImage, origMask, predMask):
	# initialize our figure
	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	# plot the original image, its mask, and the predicted mask
	ax[0].imshow(np.asarray(squeeze(origImage), dtype=int))
	ax[1].imshow((origMask))
	ax[2].imshow(collapse_mask(predMask))

	# set the titles of the subplots
	ax[0].set_title("Image")
	ax[1].set_title("Original Mask")
	ax[2].set_title("Predicted Mask")

	# set the layout of the figure and display it
	figure.tight_layout()
	figure.show()

def make_predictions(model, imagePath, mask):
	# set model to evaluation mode
	model.eval()
	#print(image)

	# turn off gradient tracking
	with torch.no_grad():
		#image = np.loadtxt(imagePath, delimiter=',', dtype=np.float32)
		image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
		cv2.imshow("image", image)
		image = np.asarray(image, dtype=np.float32)

		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()

		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)

		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		mask_image = cv2.imread(groundTruthPath, cv2.IMREAD_GRAYSCALE)
		mask_image = cv2.resize(mask_image, (128, 128))

		mask = np.zeros(mask_image.shape, dtype=np.float32)
		for index, label in enumerate(dataset.classes):
			mask[mask_image == label] = index
		mask *= 254 / len(dataset.classes)
		mask = np.asarray(mask, dtype=np.uint8)

		image = np.expand_dims(image, 0)
		image = np.expand_dims(image, 0)
		image = torch.from_numpy(image).to(config.DEVICE)

		# make the prediction, pass the results through the sigmoid
		# function, and convert the result to a NumPy array
		predMask = model(image)
		predMask = torch.sigmoid(predMask)
		predMask = predMask.cpu().numpy()
		print("Classes:")
		for i in range(predMask.shape[1]):
			print(f"{i}: {np.max(predMask[0][i])}")

		# filter out the weak predictions and convert them to integers
		# predMask = (predMask > 0.05) * 255
		# predMask = predMask.astype(np.uint8)

		# prepare a plot for visualization
		prepare_plot(orig, mask, predMask)

if __name__ == "__main__":
	# load the image and mask filepaths in a sorted manner
	imagePaths = sorted(os.path.join(config.IMAGE_DATASET_PATH,file) for file in os.listdir(config.IMAGE_DATASET_PATH))
	maskPaths = sorted(os.path.join(config.MASK_DATASET_PATH,file) for file in os.listdir(config.MASK_DATASET_PATH))

    split = train_test_split(imagePaths, maskPaths,
        test_size=config.TEST_SPLIT, random_state=42)
    (trainImages, testImages) = split[:2]
    (trainMasks, testMasks) = split[2:]

	images = np.random.choice(len(testImages), size=10)

	# load the image paths in our testing file and randomly select 10
	# image paths
	print("[INFO] loading up test image paths...")
	# imagePaths = open(config.TEST_PATHS).read().strip().split("\n")
	# imagePaths = np.random.choice(imagePaths, size=10)

	# load our model from disk and flash it to the current device
	print("[INFO] load up model...")
	unet = torch.load(config.MODEL_PATH[:-4] + "_checkpoint.pth").to(config.DEVICE)

	# iterate over the randomly selected test image paths
	for path in images:

		# make predictions and visualize the results
		make_predictions(unet, testImages[path], testMasks[path])
	input()