import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + '/../')
import config
import utils.lidar_reader as lidar_reader
from dataset import SegmentationDataset

image_path = os.path.join(config.DATASET_PATH, "images/")
mask_path = os.path.join(config.DATASET_PATH, "masks/")

image_output = os.path.join(config.DATASET_PATH, "images_png/")
mask_output = os.path.join(config.DATASET_PATH, "masks_png/")

image_output = os.path.join(config.DATASET_PATH, "../grayscale/images/")
mask_output = os.path.join(config.DATASET_PATH, "../grayscale/masks/")


for i in tqdm(range(len(os.listdir(image_path)))):
	file = np.loadtxt(os.path.join(image_path,os.listdir(image_path)[i]), delimiter=',')
	file *= (255/np.max(file))
	arr = np.asarray(file, dtype=int)
	cv2.imwrite(os.path.join(image_output,os.listdir(image_path)[i].split(".")[0]+".png"), arr)

for i in tqdm(range(len(os.listdir(mask_path)))):
	file = np.loadtxt(mask_path+os.listdir(mask_path)[i], delimiter=',')
	arr = np.asarray(file, dtype=int)
	cv2.imwrite(os.path.join(mask_output,os.listdir(mask_path)[i].split(".")[0]+".png"), arr)