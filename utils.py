import math

from scipy import interpolate
import numpy as np

import iostream

def get_rounded_multiple(to_round, round_resolution):
    if to_round % round_resolution > round_resolution / 2:
        return int(to_round / round_resolution) + 1 if to_round > 0 else int(to_round / round_resolution)
    else:
        return int(to_round / round_resolution) if to_round >= 0 else int(to_round / round_resolution) - 1

def find_coords_min_max(point_array):
    min_x = round(point_array[0][0], 2)
    max_x = round(point_array[0][0], 2)

    min_y = round(point_array[0][1], 2)
    max_y = round(point_array[0][1], 2)

    min_z = round(point_array[0][2], 2)
    max_z = round(point_array[0][2], 2)

    for i in range(1, point_array.shape[0]):
        point_array[i][0] = round(point_array[i][0], 2)
        point_array[i][1] = round(point_array[i][1], 2)
        point_array[i][2] = round(point_array[i][2], 2)

        if point_array[i][0] > max_x:
            max_x = point_array[i][0]
        elif point_array[i][0] < min_x:
            min_x = point_array[i][0]

        if point_array[i][1] > max_y:
            max_y = point_array[i][1]
        elif point_array[i][1] < min_y:
            min_y = point_array[i][1]

        if point_array[i][2] < min_z:
            min_z = point_array[i][2]
        elif point_array[i][2] > max_z:
            max_z = point_array[i][2]

    return max_x, min_x, max_y, min_y, max_z, min_z;

def point_cloud_to_grid(point_cloud, resolution, resolution_z, min_maxes = []):
    if(len(min_maxes) == 0):
        max_x, min_x, max_y, min_y, max_z, min_z = find_coords_min_max(point_cloud)
    else:
        min_x = min_maxes[0];
        max_x = min_maxes[1];
        min_y = min_maxes[2];
        max_y = min_maxes[3];


    arr = np.zeros([int(abs(max_y - min_y) / resolution) + 2, int(abs(max_x - min_x) / resolution) + 2])

    for i in range(point_cloud.shape[0]):
        x = get_rounded_multiple(point_cloud[i][0] - min_x, resolution)
        y = get_rounded_multiple(point_cloud[i][1] - min_y, resolution)
        z = point_cloud[i][2]
        if(z > arr[y][x]):
            arr[y][x] = z;

    return arr;

def grid_to_point_cloud(point_grid, resolution):
    point_cloud = np.zeros((point_grid.shape[0] * point_grid.shape[1], 3), dtype = int);
    for i in range(point_grid.shape[0]):
        for j in range(point_grid.shape[1]):
            if math.isnan(point_grid[i][j]):
                continue;
            index = i * point_grid.shape[1] + j
            point_cloud[index][0] = j * resolution;
            point_cloud[index][1] = i * resolution;
            point_cloud[index][2] = point_grid[i][j];
    return point_cloud;

def dilate(arr, to_dilate = 1, dilated = None, i = 0, j = 0):
    if(dilated == None):
        dilated = np.empty(shape = (0, 2), dtype = int);

    if(arr[i][j] == to_dilate):
        if(arr[i+1][j] == 0):
            arr[i+1][j] = to_dilate;
            dilated = np.append(dilated, [i+1, j]);

        if(arr[i][j+1] == 0):
            arr[i][j+1] = to_dilate;
            dilated = np.append(dilated, [i, j+1]);

    if(i < arr.shape[0]):
        dilate(arr, dilated, i+1, j);

    if(j < arr.shape[1]):
        dilate(arr, to_dilate, dilated, i, j+1);

def interpolate_holes(point_array, to_interpolate = None):
    nonzeros = np.nonzero(point_array)
    if to_interpolate == None:
        to_interpolate = np.argwhere(point_array == 0);
    
    fills = interpolate.griddata(nonzeros, point_array[nonzeros], to_interpolate);

    for i in range(fills.shape[0]):
        point_array[to_interpolate[i][0]][to_interpolate[i][1]] = fills[i];

    return point_array

def create_walls(point_cloud, point_grid, step = 1.0, resolution = 1.0):
    z_step = max(step, resolution);
    wall_cloud = point_cloud;
    for i in range(1, point_grid.shape[0] - 1):
        for j in range(1, point_grid.shape[1] - 1):
            minZ, d_i, d_j = get_min_surrounding(point_grid, i, j)
            if(minZ < point_grid[i][j] - z_step and minZ != -1):
                wall_cloud = np.append(wall_cloud, [[j * resolution + 0.01 * d_j, i * resolution + 0.01 * d_i, minZ]], axis = 0)
            # while(minZ < point_grid[i][j] - z_step):
            #     wall_cloud = np.append(wall_cloud, [[i * resolution, j * resolution, minZ]], axis = 0)
            #     minZ += z_step;

    return wall_cloud

def create_base(point_cloud, point_grid, resolution = 1.0, floor_z = 0):
    wall_cloud = point_cloud;
    for j in range(0, point_grid.shape[0]):
        wall_cloud = np.append(wall_cloud, [[0, j * resolution, floor_z], [(point_grid.shape[1]-1) * resolution, j * resolution, floor_z]], axis = 0)
    
    for i in range(0, point_grid.shape[1]):
        wall_cloud = np.append(wall_cloud, [[i * resolution, 0, floor_z], [i * resolution, (point_grid.shape[0]-1) * resolution, floor_z]], axis = 0)
    return wall_cloud;

def get_min_surrounding(point_grid, i, j):
    min_surrounding = -1 
    d_i = -1
    d_j = -1
    if i-1 >= 0 and (point_grid[i-1][j] < min_surrounding or min_surrounding == -1) and point_grid[i-1][j] > 0:
        min_surrounding = point_grid[i-1][j]
        d_i = -1
        d_j = 0

    if i+1 < point_grid.shape[0] and (point_grid[i+1][j] < min_surrounding or min_surrounding == -1) and point_grid[i+1][j] > 0:
        min_surrounding = point_grid[i+1][j]
        d_i = 1
        d_j = 0

    if j-1 >= 0 and (point_grid[i][j-1] < min_surrounding or min_surrounding == -1) and point_grid[i][j-1] > 0:
        min_surrounding = point_grid[i][j-1]
        d_i = 0
        d_j = -1

    if j+1 < point_grid.shape[1] and (point_grid[i][j+1] < min_surrounding or min_surrounding == -1) and point_grid[i][j+1] > 0:
        min_surrounding = point_grid[i][j+1]
        d_i = 0
        d_j = 1

    return min_surrounding, d_i, d_j
