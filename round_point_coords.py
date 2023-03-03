import numpy as np
from scipy import ndimage

from utils import *
import iostream

if __name__ == '__main__':
    arr = iostream.readInFile();

    resolution = 4.0
    resolution_z = 1

    max_x, min_x, max_y, min_y, max_z, min_z = findCoordsMinMax(arr)

    new_arr = np.zeros([int((max_y - min_y) / resolution) + 2, int((max_x - min_x) / resolution) + 2])

    for i in range(count):
        new_arr[roundMultiple(arr[i][1] - min_y, resolution)][roundMultiple(arr[i][0] - min_x, resolution)] = roundMultiple(arr[i][2], resolution_z) * resolution_z;

    new_arr = fillHoles(new_arr, 4);
    createWalls(new_arr, 1, resolution);