import argparse
import cv2
import numpy as np
import pyvista as pv

from scipy import ndimage

import utils
import iostream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct a mesh-able point cloud, given a set of ground and building points.')
    
    parser.add_argument('--ground',
                        help='Name of .txt file containing point cloud of ground')
    parser.add_argument('--buildings',
                        help='Name of .txt file containing point cloud of buildings')
    parser.add_argument('--mask',
                        help='Name of image file containing image mask of ground points')
    parser.add_argument('-o', '--output', nargs='?', const="output.stl", default="output.stl",
                        help='Name of .txt file to be output')

    parser.add_argument('-r', '--resolution', type=float, nargs='?', const=4, default=4,
                        help='Determines resolution of point grid (smaller values -> higher resolution) [Default=4]')
    parser.add_argument('-s', '--step', type=float, nargs='?', const=0, default=0,
                        help='Determines rounding of z values [Default=0]')
    parser.add_argument('-b', '--base', type=float, nargs='?', const=0, default=0,
                    help='Height of base to be generated [Default = 0]')

    args = parser.parse_args();

    resolution = args.resolution;
    resolution_z = args.step;
    base_height = args.base;

    if args.ground != None:
        ground_points = iostream.file_to_array(args.ground);
    else:
        ground_points = iostream.read_in_array("Ground file name");
    
    if args.buildings != None:
        pruned_points = iostream.file_to_array(args.buildings);
    else:
        pruned_points = iostream.read_in_array("Buildings file name");

    z_mean = np.mean(ground_points, axis=0)[2]
    z_std = np.std(ground_points, axis=0)[2]

    to_delete = []
    for i in range(ground_points.shape[0]):
        if abs(ground_points[i][2] - z_mean) > 4 * z_std:
            to_delete.append(i)
    ground_points = np.delete(ground_points, to_delete, axis = 0)

    z_mean = np.mean(pruned_points, axis=0)[2]
    z_std = np.std(pruned_points, axis=0)[2]

    to_delete = []
    for i in range(pruned_points.shape[0]):
        if abs(pruned_points[i][2] - z_mean) > 4 * z_std:
            to_delete.append(i)
    pruned_points = np.delete(pruned_points, to_delete, axis = 0)

    max_x, min_x, max_y, min_y, max_z, min_z = utils.find_coords_min_max(ground_points)

    ground_array = utils.point_cloud_to_grid(ground_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);
    pruned_array = utils.point_cloud_to_grid(pruned_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);

    if args.mask != None:
        mask = cv2.imread(args.mask)
        to_delete = []
        for i in range(pruned_array.shape[0]-1):
            for j in range(pruned_array.shape[1]-1):
                if(mask[i][j][0] == 255):
                    to_delete.append([i, j])

        for point in to_delete:
            pruned_array[point[0]][point[1]] = 0

    interpolate_mask = np.zeros([ground_array.shape[0], ground_array.shape[1]], dtype = np.byte);
    for i in range(pruned_array.shape[0]):
        for j in range(pruned_array.shape[1]):
            if pruned_array[i][j] != 0:
                interpolate_mask[i][j] = 1
    interpolate_mask = ndimage.binary_fill_holes(interpolate_mask).astype(np.byte)

    ground_array = utils.interpolate_holes(ground_array);

    combined_array = np.where(interpolate_mask == 0, ground_array, pruned_array)
    combined_array = utils.interpolate_holes(combined_array)

    pruned_cloud = utils.grid_to_point_cloud(combined_array, resolution);

    extruded_mesh = utils.point_cloud_to_mesh(pruned_cloud, base_height, [max_x, min_x, max_y, min_y, max_z, min_z])

    extruded_mesh.save(args.output)
