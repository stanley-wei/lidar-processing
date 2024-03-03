import numpy as np


def divide_ceiling(a, b):
    return -(a // -b)

def combine_bounds(bounds1, bounds2):
    '''
    Given two sets of bounds (bounds1, bounds2), combines the two:
    - For any max bound, combined max bound is max of the value for bounds1, bounds2
    - For any min bound, combined max bound is min of the value for bounds1, bounds2
    '''
    if not bounds1:
        return bounds2
    elif not bounds2:
        return bounds1

    combined_bounds = {}
    for key, item in bounds1.items():
        combined_bounds[key] = (min(bounds1[key], bounds2[key]) if key[0:3] == "min"
            else max(bounds1[key], bounds2[key]))

    return combined_bounds

def point_cloud_coord_to_grid(coord_x, coord_y, resolution, bounds):
    grid_x = round((coord_x - bounds['min_x']) / resolution)
    grid_y = round((coord_y - bounds['min_y']) / resolution)

    return grid_y, grid_x


def truncate_file_name(file_name):
    # Removes any terms preceding final / in file name
    if(len(file_name.split('/')) > 0):
        file_name = file_name.split('/')[len(file_name.split('/')) - 1]

    return file_name

