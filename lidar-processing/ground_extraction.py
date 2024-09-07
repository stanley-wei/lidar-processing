import cv2
import laspy
import math
import numpy as np
import pyvista as pv
from scipy import interpolate, spatial
from sklearn.decomposition import PCA
import sys
from timebudget import timebudget

import config
import filters
import lidar_data


@timebudget
def progressive_morphological_filter(point_cloud, cell_size=1, initial_window=2, num_steps=8,
					terrain_slope=1.2, initial_threshold=0.25, max_threshold=2.5):
	"""
	Implements a Progressive Morphological Filter for ground extraction (Zhang et al., 2003).
		- See: https://ieeexplore.ieee.org/document/1202973

	All units in meters.

	Parameters
	----------
	cell_size : int (default: 1)
		Specifies the size of the cells used when filtering. 
		Lower values correspond to higher resolution (less downsampling).

	initial_window : int (default: 2)
		Specifies the window size for morphological operations used on the first step.

	num_steps : int (default: 8)
		Specifies the number of filtering steps performed. 
		Higher values correspond to more aggressive filtering.

	terrain_slope : float (default: 1.2)
		Specifies the maximum rate of elevation increase/meter for ground points. 
		Lower values correspond to more aggressive filtering.
		Increase for hilly terrain.

	initial_threshold : float (default: 0.25)
		Specifies the initial elevation threshold used during the first step for separating
		non-ground points. 
		Higher values correspond to more aggressive filtering.

	max_threshold : float (default: 2.5)
		Specifies the maximum possible elevation threshold across all steps for separating
		non-ground points. 
		Higher values correspond to more aggressive filtering.

	"""
	max_xyz = np.max(point_cloud, axis=0)
	min_xyz = np.min(point_cloud, axis=0)

	# Create M x N x 3 grid (2D grid of points (x, y, z)) to represent surface.
	point_grid = np.zeros((math.floor((max_xyz[1]-min_xyz[1])/cell_size) + 1,
						   math.floor((max_xyz[0]-min_xyz[0])/cell_size) + 1, 3), dtype=point_cloud.dtype)
	index_grid = np.zeros((math.floor((max_xyz[1]-min_xyz[1])/cell_size) + 1,
						   math.floor((max_xyz[0]-min_xyz[0])/cell_size) + 1), dtype=int)
	index_grid.fill(-1)

	for i in range(point_cloud.shape[0]):
		y_index = math.floor((point_cloud[i][1] - min_xyz[1])/cell_size)
		x_index = math.floor((point_cloud[i][0] - min_xyz[0])/cell_size)

		if index_grid[y_index][x_index] == -1 or point_grid[y_index][x_index][2] > point_cloud[i][2]:
			point_grid[y_index][x_index] = point_cloud[i]
			index_grid[y_index][x_index] = i

	is_unused = np.ones(point_cloud.shape[0], dtype=bool)
	is_unused[np.unique(index_grid)[1:]] = False

	unused_points = np.arange(0, point_cloud.shape[0], dtype=int)[is_unused]
	unused_y = np.floor((point_cloud[unused_points, 1] - min_xyz[1])/cell_size).astype(int)
	unused_x = np.floor((point_cloud[unused_points, 0] - min_xyz[0])/cell_size).astype(int)

	unfilled_cells = np.nonzero(index_grid == -1)
	filled_cells = np.nonzero(index_grid != -1)

	# Populate empty cells via nearest-neighbor interpolation
	interpolated_values = interpolate.griddata(np.stack(filled_cells, axis=-1), point_grid[:, :, 2][filled_cells],
		np.stack(unfilled_cells, axis=-1), method='nearest')

	point_grid[:, :, 2][unfilled_cells] = interpolated_values

	grid_copy = np.array(point_grid)

	flag = np.zeros(point_grid.shape[:-1], dtype=np.uint8)

	window_sizes = [int(2 * math.pow(initial_window, i)/cell_size + 1)
						for i in range(0, num_steps)]
	elevation_threshold = initial_threshold

	# Filtering step
	nonground_indices = np.zeros(0, dtype=int)
	elevation_grid = np.asarray(point_grid[:, :, 2])
	for i in range(1, len(window_sizes)):
		kernel = np.ones((window_sizes[i], window_sizes[i]), dtype=np.uint8)

		# Apply morphological operations
		filtered_grid = np.asarray(elevation_grid)
		filtered_grid = cv2.erode(filtered_grid, kernel, iterations=1)
		filtered_grid = cv2.dilate(filtered_grid, kernel, iterations=1)

		# Mark points exceeding elevation threshold as non-ground
		flag[np.nonzero(elevation_grid - filtered_grid > elevation_threshold)] = i

		unused_indices = point_cloud[unused_points, 2] - filtered_grid[y_index, x_index] > elevation_threshold
		nonground_indices = np.concatenate((nonground_indices, unused_points[unused_indices]), dtype=int)

		unused_indices = np.invert(unused_indices)
		unused_points = unused_points[unused_indices]
		unused_y = unused_y[unused_indices]
		unused_x = unused_x[unused_indices]

		# Compute new elevation threshold
		elevation_threshold = min(
			terrain_slope * (window_sizes[i]-window_sizes[i-1]) * cell_size + initial_threshold, 
			max_threshold)

		elevation_grid = filtered_grid


	# Remove interpolated cells & separate ground vs. non-ground points
	keep_cells = (index_grid != -1)
	flag_zeros = (flag == 0)
	flag_nonzeros = np.invert(flag_zeros)

	ground_points = index_grid[np.logical_and(keep_cells, flag_zeros)]
	non_ground = index_grid[np.logical_and(keep_cells, flag_nonzeros)]

	classifications = np.zeros(point_cloud.shape[0], dtype=int)
	classifications[np.concatenate((ground_points, unused_points))] = config.GROUND
	classifications[np.concatenate((non_ground, nonground_indices))] = config.UNASSIGNED

	return classifications


if __name__ == "__main__":
	las_data = laspy.open(sys.argv[1]).read()
	points = lidar_data.PointCloud(np.asarray(las_data.xyz), np.asarray(las_data.classification))

	# Feet to meters
	# points *= 0.3048

	points = filters.voxel_filter(points, resolution=1.0)
	points = filters.remove_statistical_outliers(points)

	# Extract ground surface
	# classifications = progressive_morphological_filter(points.point_cloud)

	# Write to LiDAR file
	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = points.point_cloud
	classified_las.classification = points.classification
	classified_las.write("classified.laz")
