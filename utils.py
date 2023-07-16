import math

from scipy import interpolate
import numpy as np
import pyvista as pv

import iostream

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
        x = round((point_cloud[i][0] - min_x) / resolution)
        y = round((point_cloud[i][1] - min_y) / resolution)
        z = point_cloud[i][2]
        if(z > arr[y][x]):
            arr[y][x] = z;

    return arr;

def grid_to_point_cloud(point_grid, resolution):
    point_cloud = np.zeros((0, 3), dtype = int);
    to_delete = []
    for i in range(point_grid.shape[0]):
        for j in range(point_grid.shape[1]):
            if not math.isnan(point_grid[i][j]):
                to_append = [[j * resolution, i * resolution, point_grid[i][j]]]
                point_cloud = np.append(point_cloud, to_append, axis=0)
    return point_cloud;

def interpolate_holes(point_array):
    nonzeros = np.nonzero(point_array)
    grid_x, grid_y = np.meshgrid(range(point_array.shape[0]), range(point_array.shape[1]), indexing='ij')

    point_array = interpolate.griddata(nonzeros, point_array[nonzeros], (grid_x, grid_y));

    holes = np.concatenate((np.where(np.isnan(point_array)), np.where(point_array == 0)), axis=1)
    if holes.any():
        nonzeros = np.nonzero(point_array)
        point_array = interpolate.griddata(nonzeros, point_array[nonzeros], (grid_x, grid_y), method='nearest')

    return point_array

def point_cloud_to_mesh(point_cloud, base_height = 0, min_maxes = []):
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

    plane = pv.Plane(
                center = (surface.center[0], surface.center[1], min_z - base_height),
                direction = (0, 0, -1.0),
                i_size = max_x - min_x,
                j_size = max_y - min_y)

    extruded_mesh = surface.extrude_trim((0, 0, -1.0), plane)
    return extruded_mesh
