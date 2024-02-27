import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + '/../')
import config

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Filter dataset points')

	parser.add_argument('-t', '--threshold', type=int, nargs='?', const=0.05, default=0.05,
	                help='Proportion of building points needed to keep a file [Default: 0.05]')
	parser.add_argument('-f', '--filter', type=int, nargs='?', const=0.1, default=0.1,
	                help='Proportion of irrelevant points needed to remove a file [Default: 0.1]')

	args = parser.parse_args();

	ignore_image_path = os.path.join(config.DATASET_PATH, "images_ignore/")
	ignore_mask_path = os.path.join(config.DATASET_PATH, "masks_ignore/")

	num_ignored = 0
	dataset_size = len(os.listdir(config.MASK_DATASET_PATH))

	to_rename = []

	for i in tqdm(range(0, len(os.listdir(config.MASK_DATASET_PATH)))):
		file_name = os.listdir(config.MASK_DATASET_PATH)[i]
		arr = cv2.imread(config.MASK_DATASET_PATH + file_name, cv2.IMREAD_GRAYSCALE)

		if (arr.shape[0] < config.INPUT_WIDTH or arr.shape[1] < config.INPUT_HEIGHT
			# or np.count_nonzero(arr > 6) > (args.filter * (arr.shape[0] * arr.shape[1])) 
			# or np.count_nonzero(arr == 6) < (args.threshold * (arr.shape[0] * arr.shape[1])))
			):

			to_rename.append(file_name)
			num_ignored += 1

	for i in tqdm(range(0, len(to_rename))):
		file_name = to_rename[i]

		os.rename(config.IMAGE_DATASET_PATH + file_name, ignore_image_path + file_name)
		os.rename(config.MASK_DATASET_PATH + file_name, ignore_mask_path + file_name)

	print(f"{num_ignored} / {dataset_size}")
