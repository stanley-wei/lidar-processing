import shutil
import cv2
import numpy as np
from scipy import ndimage
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    else:
        file_name = input("File name: ")
    # file_name = "lowdensity2_003_04.txt"
    file = open(file_name, 'r')
    with open(file_name) as f:
        count = sum(1 for _ in f)

    arr = np.loadtxt(file_name, delimiter=" ", dtype=float)
    arr.reshape(count, -1)
    arr = np.delete(arr, list(range(3, arr.shape[1])), axis = 1)

    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    min_z = 0
    max_z = 0

    for i in range(count):
        arr[i][0] = round(arr[i][0], 2)
        arr[i][1] = round(arr[i][1], 2)
        arr[i][2] = round(arr[i][2], 2)

        if i == 0:
            min_x = arr[0][0]
            max_x = arr[0][0]

            min_y = arr[0][1]
            max_y = arr[0][1]

            min_z = arr[0][2]
            max_z = arr[0][2]
        else:
            if arr[i][0] > max_x:
                max_x = arr[i][0]
            elif arr[i][0] < min_x:
                min_x = arr[i][0]

            if arr[i][1] > max_y:
                max_y = arr[i][1]
            elif arr[i][1] < min_y:
                min_y = arr[i][1]

            if arr[i][2] < min_z:
                min_z = arr[i][2]
            elif arr[i][2] > max_z:
                max_z = arr[i][2]

    new_arr = np.zeros([int((max_y - min_y)/2.5) + 1, int((max_x - min_x)/2.5) + 1], dtype = np.uint8)
    z_scale = 255 / (max_z - min_z)

    newX = 0;
    newY = 0;
    for i in range(count):
        newX = int((arr[i][0] - min_x)/2.5)
        newY = int((arr[i][1] - min_y)/2.5)
        newZ = int(z_scale * (arr[i][2] - min_z))
        new_arr[newY][newX] = newZ

    #new_arr = ndimage.binary_fill_holes(new_arr).astype(np.uint8)
    cv2.imwrite(file_name.split('.')[0] + ".jpg", new_arr)

    print(new_arr)
    print(z_scale)
    print([min_x, max_x, min_y, max_y, min_z, max_z])

    file.close()