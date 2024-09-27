import argparse
import cv2
import laspy
import logging
import numpy as np
import pyvista as pv
from scipy import ndimage
import sys
from tqdm import tqdm

from ..data import PointCloud, PointGrid


def request_mask(point_grid, mask_name, generate_mask):
    '''
    Obtains a mask from user arguments, or generates a new one
    if none was specified.
    '''
    if not generate_mask:
        mask_image = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
    else:
        point_grid.generate_image(file_name=mask_name)

        print(f"Mask-to-be-modified written to file: {mask_name}")
        mask_file = input(f"Type file name of mask when complete (press Enter to use {mask_name}): ").rstrip()
        
        if mask_file == "":
            mask_file = mask_name
        mask_image = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    if(mask_image.shape[0:2] != point_grid.shape):
        print(f"Mask image size is different from array size: mask {mask_image.shape} vs array {point_grid.shape}")
        sys.exit(1)

    return mask_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Given a classified LiDAR .las/.laz file, \
      constructs a 3D mesh after cleaning & interpolating LiDAR point cloud. Supports filtering of mesh output via image masks.")
    
    parser.add_argument('file',
                        help='Name of .las/.laz file containing classified point cloud')
    parser.add_argument('output', nargs='?', const="output.stl", default="output.stl",
                        help='Name of .stl file to be output [Default: output.stl]')


    interpolation_options = parser.add_argument_group("interpolation options")
    interpolation_options.add_argument('-r', '--resolution', type=float, nargs='?', const=5, default=5,
                        help='Determines resolution of point grid (smaller values -> higher resolution) [Default=4]')
    interpolation_options.add_argument('-b', '--base', type=float, nargs='?', const=0, default=0,
                    help='Height of base to be generated [Default = 0]')
    interpolation_options.add_argument('--disable-discretize', dest='disable_discretize', action='store_true',
                        help='Prevents point cloud from being converted to discretized grid')
    interpolation_options.add_argument('--include-unclassified', dest='incl_unclassified', action='store_true',
                        help='Include buildings not classified as ground or building in meshing')


    masking_options = parser.add_argument_group("masking options")
    masking_options.add_argument('--mask',
                        help='Name of image file containing image mask of model')
    masking_options.add_argument('--generate-mask', dest='generate_mask', action='store_true',
                        help='Name of image mask file to be generated')

    masking_options.add_argument('--tree-mask', dest='tree_mask',
                        help='Name of image file containing image mask for tree')
    masking_options.add_argument('--generate-tree-mask', dest='generate_tree_mask', action='store_true',
                        help='Name of tree mask file to be generated')


    parser.add_argument('--debug', dest='enable_logging', action='store_true',
                        help='Display debug/info messages [Default = False]')
    parser.add_argument('--log-file', dest='log_file',
                        help='Name of file to which logging messages will be written')

    args = parser.parse_args();

    if args.enable_logging:
        logging.basicConfig(level=logging.DEBUG)
    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', format='%(name)s: %(levelname)s: %(message)s')

    resolution = args.resolution;


    # Uses the Laspy library to read in LiDAR data
    if args.file == None:
        args.file = input(".las/.laz file name:")

    print("Reading LAS/LAZ file")
    las_file = laspy.open(args.file)
    las_points = las_file.read()

    las_xyz = las_points.xyz
    las_classifications = las_points.classification


    # Separate LiDAR data by classification
    # See: https://desktop.arcgis.com/en/arcmap/latest/manage-data/las-dataset/lidar-point-classification.htm
    ground_points = PointCloud(las_xyz[np.where(las_classifications == 2)])
    ground_points.remove_outliers(num_stdev = 2)

    building_points = PointCloud(las_xyz[np.where(las_classifications == 6)])
    z_mean, z_std = building_points.remove_outliers(num_stdev=2)

    if args.incl_unclassified:
        extra_points = PointCloud(las_xyz[np.where(np.isin(las_classifications, [2, 5, 6], invert=True))])
        extra_points.remove_outliers(num_stdev=0.5)

        building_points = PointCloud.combine(building_points, extra_points)

    ground_array = PointGrid.from_point_cloud(ground_points, resolution, ground_points.bounds);
    pruned_array = PointGrid.from_point_cloud(building_points, resolution, ground_points.bounds);


    # Interpolate ground, non-ground arrays & combine
    print("Beginning interpolation")

    ground_array.interpolate_holes()

    interpolated_array, interpolate_mask = PointGrid.overlay(base_grid=ground_array, overlaying_grid=pruned_array)
    interpolated_array.interpolate_holes()

    if not args.disable_discretize: # Take grid, rather than raw point cloud, as source of non-ground points
        output_cloud = interpolated_array.to_point_cloud(invert_xy=True)
    else:
        output_cloud = PointCloud.combine(building_points, ground_points)

    # Generate and apply masks (if applicable)
    excluded_points = [] # Points to remove during final meshing process

    if args.mask:
        print("Applying mask")
        mask_image = request_mask(interpolated_array, args.mask, args.generate_mask)
        excluded_points = output_cloud.apply_mask(mask_image, interpolated_array.resolution)

    if args.tree_mask:
        print("Applying tree mask")
        tree_points = PointCloud(las_xyz[np.where(las_classifications == 5)])
        z_mean, z_std = tree_points.remove_outliers(num_stdev=4)

        tree_array = PointGrid.from_point_cloud(tree_points, resolution, ground_points.bounds);
        tree_cloud = tree_array.to_point_cloud(invert_xy=True, ignore_zeros=True)

        tree_mask = request_mask(tree_array, args.tree_mask, args.generate_tree_mask)
        excluded_points = excluded_points + tree_cloud.apply_mask(tree_mask, tree_array.resolution, start_index=output_cloud.shape[0])
        output_cloud = PointCloud.combine(output_cloud, tree_cloud)


    # Convert point cloud to mesh & output
    print("Generating mesh")
    extruded_mesh = output_cloud.generate_mesh(excluded_points, base_height=args.base)

    extruded_mesh.save(args.output)
    print("Mesh saved to file: " + args.output)
