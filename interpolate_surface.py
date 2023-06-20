import argparse
import numpy as np

import utils
import iostream

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct a mesh-able point cloud, given a set of ground and building points.')
    
    parser.add_argument('--ground',
                        help='Name of .txt file containing point cloud of ground')
    parser.add_argument('--buildings',
                        help='Name of .txt file containing point cloud of buildings')
    parser.add_argument('-o', '--output', nargs='?', const="output.txt", default="output.txt",
                        help='Name of .txt file to be output')

    parser.add_argument('-r', '--resolution', type=int, nargs='?', const=4, default=4,
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

    z_mean = np.mean(pruned_points, axis=0)[2]
    z_std = np.std(pruned_points, axis=0)[2]

    to_delete = []
    for i in range(pruned_points.shape[0]):
        if(pruned_points[i][2] > z_mean + 4 * z_std):
            to_delete.append(i)
            
    pruned_points = np.delete(pruned_points, to_delete, axis = 0)

    max_x, min_x, max_y, min_y, max_z, min_z = utils.find_coords_min_max(ground_points)

    # orig_array = utils.point_cloud_to_grid(orig_points, resolution, resolution_z, [min_x, min_y]);
    ground_array = utils.point_cloud_to_grid(ground_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);
    pruned_array = utils.point_cloud_to_grid(pruned_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);

    new_cloud = np.zeros([ground_array.shape[0], ground_array.shape[1]], dtype = np.byte);

    for i in range(ground_array.shape[0]):
        for j in range(ground_array.shape[1]):
            if(pruned_array[i][j] != 0):
                new_cloud[i][j] = ord('b');
            elif(ground_array[i][j] != 0):
                new_cloud[i][j] = ord('g');
    
    zeros = np.argwhere(new_cloud == 0);
    ground_interpolate = np.empty(shape = (0, 2), dtype = int)
    other_interpolate = np.empty(shape = (0, 2), dtype = int)

    no_change = False;
    while(no_change == False):
        no_change = True;
        for i in range(new_cloud.shape[0]):
            for j in range(new_cloud.shape[1]):
                if(new_cloud[i][j] == 0):
                    if((i != 0 and new_cloud[i-1][j] == ord('g')) or (i < new_cloud.shape[0]-1 and new_cloud[i+1][j] == ord('g')) or (j != 0 and new_cloud[i][j-1] == ord('g')) or (j < new_cloud.shape[1]-1 and new_cloud[i][j+1] == ord('g'))):
                        new_cloud[i][j] = ord('g');
                        no_change = False;
                        ground_interpolate = np.append(ground_interpolate, [[i, j]], axis=0);

    #ground_interpolate = utils.dilate(new_cloud, ord('g'));
    
    fill_zeros = np.argwhere(new_cloud == 0);
    for i in range(fill_zeros.shape[0]):
        new_cloud[fill_zeros[i][0]][fill_zeros[i][1]] = ord('b');
        other_interpolate = np.append(other_interpolate, [[fill_zeros[i][0], fill_zeros[i][1]]], axis=0);

    ground_array = utils.interpolate_holes(ground_array);

    for i in range(ground_interpolate.shape[0]):
        pruned_array[ground_interpolate[i][0]][ground_interpolate[i][1]] = ground_array[ground_interpolate[i][0]][ground_interpolate[i][1]];
        
    pruned_array = utils.interpolate_holes(pruned_array);

    pruned_cloud = utils.grid_to_point_cloud(pruned_array, resolution);

    wall_cloud = utils.create_walls(pruned_cloud, pruned_array, resolution_z, resolution);

    to_delete = []
    for i in range(wall_cloud.shape[0]):
        if(wall_cloud[i][2] == 0):
            to_delete.append(i)

    wall_cloud = np.delete(wall_cloud, to_delete, axis = 0)

    wall_cloud = utils.create_base(wall_cloud, pruned_array, resolution, min_z - base_height)

    iostream.write_to_file(wall_cloud, args.output);
