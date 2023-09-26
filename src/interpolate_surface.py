import argparse
import asyncio
import cv2
import laspy
import logging
import numpy as np
import pyvista as pv
from scipy import ndimage
from tqdm import tqdm

import utils
import iostream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct a mesh-able point cloud, given a set of ground and building points.')
    
    parser.add_argument('--file',
                        help='Name of .las/.laz file containing point cloud of ground')
    parser.add_argument('-o', '--output', nargs='?', const="output.stl", default="output.stl",
                        help='Name of .txt file to be output')

    parser.add_argument('--mask',
                        help='Name of image file containing image mask of model')
    parser.add_argument('--generate-mask', dest='generate_mask', action='store_true',
                        help='Name of image mask file to be generated')

    parser.add_argument('-r', '--resolution', type=float, nargs='?', const=5, default=5,
                        help='Determines resolution of point grid (smaller values -> higher resolution) [Default=4]')
    parser.add_argument('-s', '--step', type=float, nargs='?', const=0, default=0,
                        help='Determines rounding of z values [Default=0]')
    parser.add_argument('-b', '--base', type=float, nargs='?', const=0, default=0,
                    help='Height of base to be generated [Default = 0]')

    parser.add_argument('--debug', dest='enable_logging', action='store_true',
                        help='Display debug/info messages [Default = True]')
    parser.add_argument('--log-file', dest='log_file',
                        help='Name of file to which logging messages will be written')

    parser.set_defaults(enable_logging=True, generate_mask=False)

    args = parser.parse_args();

    if args.enable_logging:
        logging.basicConfig(level=logging.DEBUG)
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', format='%(name)s: %(levelname)s: %(message)s')

    resolution = args.resolution;
    resolution_z = args.step;
    base_height = args.base;

    if args.file == None:
        args.file = input(".las/.laz file name:")

    logging.info("Reading LAS/LAZ file")
    las_file = laspy.open(args.file)
    las_points = las_file.read()

    las_xyz = las_points.xyz
    las_classifications = las_points.classification

    ground_points = las_xyz[np.where(las_classifications == 2)]
    pruned_points = las_xyz[np.where(las_classifications == 6)]
    other_not_tree_points = las_xyz[np.where(np.isin(las_classifications, [2, 5, 6], invert=True))]

    z_mean = np.mean(ground_points, axis=0)[2]
    z_std = np.std(ground_points, axis=0)[2]

    to_delete = np.where(abs(ground_points[2] - z_mean) > 4 * z_std)
    ground_points = np.delete(ground_points, to_delete, axis = 0)

    z_mean = np.mean(pruned_points, axis=0)[2]
    z_std = np.std(pruned_points, axis=0)[2]

    to_delete = np.where(pruned_points[2] - z_mean > 4 * z_std)
    pruned_points = np.delete(pruned_points, to_delete, axis = 0)

    max_x, min_x, max_y, min_y, max_z, min_z = utils.find_coords_min_max(ground_points)

    ground_array = utils.point_cloud_to_grid(ground_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);
    pruned_array = utils.point_cloud_to_grid(pruned_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);

    logging.info("Beginning interpolation")
    interpolate_mask = np.zeros([ground_array.shape[0], ground_array.shape[1]], dtype = np.byte);
    for i in range(pruned_array.shape[0]):
        for j in range(pruned_array.shape[1]):
            if pruned_array[i][j] != 0:
                interpolate_mask[i][j] = 1
    interpolate_mask = ndimage.binary_fill_holes(interpolate_mask).astype(np.byte)

    ground_array = utils.interpolate_holes(ground_array);

    combined_array = np.where(interpolate_mask == 0, ground_array, pruned_array)
    interpolated_array = utils.interpolate_holes(combined_array)

    to_remove = []
    final_array = np.flip(interpolated_array, 0)

    if args.mask != None:
        logging.info("Applying mask")
        if not args.generate_mask:
            mask_image = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        else:
            utils.create_mask(final_array, args.mask)
            print("Mask-to-be-modified written to file: " + args.mask)
            mask_file = input("Type file name of mask when complete: ")
            mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        if(mask_image.shape[0:2] != final_array.shape):
            print("Mask image must be same size as array")
            sys.exit(1)

        pruned_cloud, to_remove = utils.masked_grid_to_point_cloud(final_array, mask_image, resolution);

    else:
        pruned_cloud = utils.grid_to_point_cloud(final_array, resolution);

    logging.info("Generating mesh")
    extruded_mesh = utils.point_cloud_to_mesh(pruned_cloud, to_remove, base_height=base_height)

    extruded_mesh.save(args.output)
    print("Mesh saved to file: " + args.output)
