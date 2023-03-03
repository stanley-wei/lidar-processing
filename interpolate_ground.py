import numpy as np
from scipy import ndimage, interpolate

from utils import *

import iostream

if __name__ == '__main__':
    # if len(sys.argv) > 1:
    #     file_name = sys.argv[1]
    # else:
    #     file_name = input("File name: ")

    file_name = "Input/Royce_Hall_Ground.txt"

    arr = iostream.fileToNumpy(file_name)

    resolution = 4.0
    resolution_z = 1

    new_arr = pointCloudToGrid(arr, resolution, resolution_z);

    new_arr = interpolateHoles(new_arr)

    iostream.writeToFile(new_arr, "filled_ground.txt", resolution)
    