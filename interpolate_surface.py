import utils
import iostream

if __name__ == "__main__":
    # orig_points = iostream.read_in_array("Original file name");
    ground_points = iostream.read_in_array("Ground file name");
    pruned_points = iostream.read_in_array("Pruned file name");

    resolution = 4;
    resolution_z = 1;

    max_x, min_x, max_y, min_y, max_z, min_z = utils.find_coords_min_max(ground_points)

    # orig_array = utils.point_cloud_to_grid(orig_points, resolution, resolution_z, [min_x, min_y]);
    ground_array = utils.point_cloud_to_grid(ground_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);
    pruned_array = utils.point_cloud_to_grid(pruned_points, resolution, resolution_z, [min_x, max_x, min_y, max_y]);

    ground_array = utils.interpolate_holes(ground_array);

    for i in range(pruned_array.shape[0]):
        for j in range(pruned_array.shape[1]):
            if(pruned_array[i][j] == 0):
                pruned_array[i][j] = ground_array[i][j];

    utils.create_walls(pruned_array, resolution_z, resolution);
