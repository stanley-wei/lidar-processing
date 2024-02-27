import asyncio
import laspy
import laszip
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + '/../')
import config
import utils.lidar_reader as lidar_reader

dataset_path = config.DATASET_PATH + '/chicago/'

for i in tqdm(range(0, len(os.listdir(dataset_path)))):
	try:
		file = lidar_reader.LazFile(f"{dataset_path}{os.listdir(dataset_path)[i]}")
		file.split_laz(config.LIDAR_RESOLUTION)

	except laszip.laszip.LaszipError:
		print(f'laszip - {i}')
		continue

	except laspy.errors.LaspyException:
		print(f'laspy - {i}')
		continue
