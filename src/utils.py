import cv2
import math
import numpy as np
import pyvista as pv
from scipy import interpolate
from tqdm import tqdm

import iostream

def find_coords_min_max(point_array):
    minimums = np.min(point_array, axis = 0)
    maximums = np.max(point_array, axis = 0)

    return maximums[0], minimums[0], maximums[1], minimums[1], maximums[2], minimums[2]

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
        x = round((point_cloud[i][0] - min_x) / resolution)
        y = round((point_cloud[i][1] - min_y) / resolution)
        z = point_cloud[i][2]
        if(z > arr[y][x]):
            arr[y][x] = z;

    return arr;

def grid_to_point_cloud(point_grid, resolution):
    point_cloud = np.zeros((point_grid.shape[0] * point_grid.shape[1], 3), dtype = int);
    index = 0
    i_multi = 0
    for i in tqdm(range(point_grid.shape[0])):
        j_multi = 0
        for j in range(point_grid.shape[1]):
            if not math.isnan(point_grid[i][j]):
                point_cloud[index] = [i_multi, j_multi, point_grid[i][j]]
            j_multi += resolution
            index += 1
        i_multi += resolution
    point_cloud = point_cloud[point_cloud.any(axis=1)]
    return point_cloud;

def masked_grid_to_point_cloud(point_grid, mask_image, resolution):
    point_cloud = np.zeros((point_grid.shape[0] * point_grid.shape[1], 3), dtype = int);

    contours, hierarchy = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_stack = np.vstack(contours).reshape(-1, 2)

    to_remove = []
    index = 0
    i_multi = 0
    for i in tqdm(range(point_grid.shape[0])):
        j_multi = 0
        for j in range(point_grid.shape[1]):
            if not math.isnan(point_grid[i][j]):
                point_cloud[index] = [i_multi, j_multi, point_grid[i][j]]
                if mask_image[i][j] == 0:
                    to_remove.append(index)
                index += 1
            j_multi += resolution
        i_multi += resolution
    point_cloud = point_cloud[point_cloud.any(axis=1)]

    point_cloud = np.delete(point_cloud, range(index, len(point_cloud)-1), axis=0)

    return point_cloud, to_remove

def interpolate_holes(point_array):
    nonzeros = np.nonzero(point_array)
    not_nan_or_zero = np.asarray(nonzeros)[:, np.where(~np.isnan(point_array[nonzeros]))[0]]
    grid_x, grid_y = np.meshgrid(range(point_array.shape[0]), range(point_array.shape[1]), indexing='ij')

    point_array = interpolate.griddata((not_nan_or_zero[0], not_nan_or_zero[1]), point_array[(not_nan_or_zero[0], not_nan_or_zero[1])], (grid_x, grid_y));

    holes = np.concatenate((np.where(np.isnan(point_array)), np.where(point_array == 0)), axis=1)
    if holes.any():
        nonzeros = np.nonzero(point_array)
        not_nan_or_zero = np.asarray(nonzeros)[:, np.where(~np.isnan(point_array[nonzeros]))[0]]
        point_array = interpolate.griddata((not_nan_or_zero[0], not_nan_or_zero[1]), point_array[(not_nan_or_zero[0], not_nan_or_zero[1])], (grid_x, grid_y), method='nearest');

    return point_array

def point_cloud_to_mesh(point_cloud, to_remove = [], base_height = 0, min_maxes = []):
    if not min_maxes:
        min_maxes = find_coords_min_max(point_cloud);

    max_x = min_maxes[0]
    min_x = min_maxes[1]
    max_y = min_maxes[2]
    min_y = min_maxes[3]
    max_z = min_maxes[4]
    min_z = min_maxes[5]

    pv_cloud = pv.PolyData(point_cloud)
    surface = pv_cloud.delaunay_2d()

    if to_remove:
        surface, indices = surface.remove_points(to_remove)

    plane = pv.Plane(
                center = (surface.center[0], surface.center[1], min_z - base_height),
                direction = (0, 0, -1.0),
                i_size = max_x - min_x,
                j_size = max_y - min_y)

    extruded_mesh = surface.extrude_trim((0, 0, -1.0), plane)
    return extruded_mesh

def create_mask(array, file_name):
    max_z = np.max(array)
    min_z = np.min(array)

    array = (array - min_z) * (255.0 / (max_z - min_z))

    integer_array = np.asarray(array, dtype=int)
    cv2.imwrite(file_name, integer_array)
