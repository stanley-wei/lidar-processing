import cv2
import laspy
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(__file__) + '/../')
import config

class LazFile:
    def __init__(self, file_name):
        self.file_name = file_name
        self.laz_file = laspy.open(file_name)
        self.point_cloud = self.laz_file.read()

        point_array = self.point_cloud.xyz
        classifications = self.point_cloud.classification
        self.points = np.concatenate((point_array, np.asarray(classifications).reshape(-1, 1)), axis=1)

        self.real_width = self.laz_file.header.x_max - self.laz_file.header.x_min
        self.real_height = self.laz_file.header.y_max - self.laz_file.header.y_min

        self.split_width = config.DIVIDE_WIDTH
        self.split_height = config.DIVIDE_HEIGHT
        self.split_resolution = config.LIDAR_RESOLUTION

    def __getitem__(self, key):
        grid_index = int(self.real_width / self.split_width)

        min_x = self.split_width * int(key % grid_index) + self.laz_file.header.x_min
        min_y = self.split_height * int(key / grid_index) + self.laz_file.header.y_min

        selected_points = self.points[((self.points[:, 0] >= min_x) 
                            & (self.points[:, 0] < min_x + self.split_width)
                            & (self.points[:, 1] >= min_y)
                            & (self.points[:, 1] < min_y + self.split_height))]

        return selected_points

    @property
    def shape(self):
        grid_width = int(self.real_width / self.split_width)
        grid_height = int(self.real_height / self.split_height)

        return (grid_width, grid_height)

    def split_laz(self, resolution):
        grid_width = int(self.real_width / self.real_height)
        grid_height = int(self.real_height / self.split_height)

        file_path = config.TRAIN_PATH

        num_images = grid_height * grid_width
        for i in range(num_images):
            image, mask = discretize(self[i], self.split_width, self.split_height, resolution)

            file_name = self.file_name.split(".")[-2].split("/")[-1]+f"_{i}"

            np.savetxt(f"{file_path}/images/{file_name}.csv", image, delimiter=",")
            np.savetxt(f"{file_path}/masks/{file_name}.csv", mask, delimiter=",")

def discretize(array, width, height, resolution):
    z_mean = np.mean(array, axis=0)[2]
    z_std = np.std(array, axis=0)[2]

    to_delete = []
    for i in range(array.shape[0]):
        if abs(array[i][2] - z_mean) > 4 * z_std:
            to_delete.append(i)
    array = np.delete(array, to_delete, axis = 0)

    minimums = np.min(array, axis = 0)
    min_x = minimums[0]
    min_y = minimums[1]
    min_z = minimums[2]

    array[:,0] -= min_x
    array[:,1] -= min_y
    array[:,2] -= min_z
    array[:, 0:2] /= resolution

    shape = np.asarray((np.asarray([width, height]) / resolution) + 1, dtype=int)
    base_arr = np.zeros(shape, dtype = float)
    mask_arr = np.zeros(shape, dtype = int)

    for i in range(array.shape[0]):
        newX = int(array[i][0])
        newY = int(array[i][1])
        newZ = array[i][2]

        if newZ > base_arr[newY][newX]:
            base_arr[newY][newX] = newZ + 1
            mask_arr[newY][newX] = array[i][3]

    return base_arr, mask_arr
