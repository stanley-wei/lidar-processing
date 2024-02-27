import cv2
import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import random

def get_unique_values(maskPath, data_type):
	if data_type == "csv":
		mask = np.loadtxt(maskPath, delimiter=',')
	elif data_type == "image":
		mask = np.asarray(cv2.imread(maskPath))
	return np.unique(mask)

class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms, data_type="image"):
		# store the image and mask filepaths, and augmentation transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
		self.data_type = data_type

		unique_classes = []
		for maskPath in self.maskPaths:
			mask_classes = get_unique_values(maskPath, self.data_type)
			unique_classes.append(mask_classes)

		self.classes = list(sorted(np.unique(np.concatenate(unique_classes), axis=0).tolist()))
		print(self.classes)
		print(list(enumerate(self.classes)))

	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)

	def __getitem__(self, idx):
		if self.data_type == "csv":
			image = np.loadtxt(self.imagePaths[idx], delimiter=',', dtype=np.float32)
			mask_image = np.loadtxt(self.maskPaths[idx], delimiter=',', dtype=np.uint8)
		elif self.data_type == "image":
			image = cv2.imread(self.imagePaths[idx], cv2.IMREAD_GRAYSCALE)
			mask_image = np.asarray(cv2.imread(self.maskPaths[idx], cv2.IMREAD_GRAYSCALE))

		mask = np.zeros(mask_image.shape, dtype=np.uint8)
		for index, label in enumerate(self.classes):
			mask[mask_image == label] = index

		# check to see if we are applying any transformations
		if self.transforms is not None:
			transformed = self.transforms(image=image, mask=mask)

			# apply the transformations to both image and its mask
			image = transformed['image']
			mask = transformed['mask']

		# if bool(random.getrandbits(1)):
		# 	image = np.flip(image, axis=0)
		# 	mask = np.flip(mask, axis=0)

		# if bool(random.getrandbits(1)):
		# 	image = np.flip(image, axis=1)
		# 	mask = np.flip(mask, axis=1)

		# num_rot = random.randint(0,3)
		# image = np.rot90(image, num_rot)
		# mask = np.rot90(mask, num_rot)

		image = transforms.ToTensor()(image)

		# mask = cv2.merge(masks)
		mask = np.array(mask)
		mask = torch.from_numpy(mask).long()

		# return a tuple of the image and its mask
		return (image, mask)

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
