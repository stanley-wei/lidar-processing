import numpy as np
from scipy import ndimage

from utils import *
import iostream

if __name__ == '__main__':
    arr = iostream.readInFile();

    resolution = 4.0
    resolution_z = 1

    new_arr = point_cloud_to_grid(arr, resolution, resolution_z);

    create_walls(new_arr, 1, resolution);

