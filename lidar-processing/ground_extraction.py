import laspy
import math
import numpy as np
import pyvista as pv
from scipy import interpolate, spatial
from sklearn.decomposition import PCA
import sys
from timebudget import timebudget

import cv2
import normals

class ProgressiveMorphologicalFilter:
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

	def __init__(self, cell_size=1, initial_window=2, num_steps=8,
					terrain_slope=0.5, initial_threshold=0.25, max_threshold=2.5):
		self.cell_size = cell_size
		self.initial_window = initial_window
		self.num_steps = num_steps
		self.terrain_slope = terrain_slope
		self.initial_threshold = initial_threshold
		self.max_threshold = max_threshold

	@timebudget
	def extract_ground(self, point_cloud):
		max_xyz = np.max(point_cloud, axis=0)
		min_xyz = np.min(point_cloud, axis=0)

		# Create M x N x 3 grid (2D grid of points (x, y, z)) to represent surface.
		point_grid = np.zeros((math.floor((max_xyz[1]-min_xyz[1])/self.cell_size) + 1,
							   math.floor((max_xyz[0]-min_xyz[0])/self.cell_size) + 1, 3), dtype=point_cloud.dtype)

		point_grid[:, :, 2] = -1

		for i in range(point_cloud.shape[0]):
			y_index = math.floor((point_cloud[i][1] - min_xyz[1])/self.cell_size)
			x_index = math.floor((point_cloud[i][0] - min_xyz[0])/self.cell_size)

			if point_grid[y_index][x_index][2] == -1 or point_grid[y_index][x_index][2] > point_cloud[i][2]:
				point_grid[y_index][x_index] = point_cloud[i]

		unfilled_cells = np.nonzero(point_grid[:, :, 2] == -1)
		filled_cells = np.nonzero(point_grid[:, :, 2] != -1)

		# Populate empty cells via nearest-neighbor interpolation
		interpolated_values = interpolate.griddata(np.stack(filled_cells, axis=-1), point_grid[:, :, 2][filled_cells],
			np.stack(unfilled_cells, axis=-1), method='nearest')

		point_grid[:, :, 2][unfilled_cells] = interpolated_values

		grid_copy = np.array(point_grid)

		flag = np.zeros(point_grid.shape[:-1], dtype=np.uint8)

		window_sizes = [int(2 * math.pow(self.initial_window, i)/self.cell_size + 1)
							for i in range(0, self.num_steps)]
		elevation_threshold = self.initial_threshold

		# Filtering step
		elevation_grid = np.asarray(point_grid[:, :, 2])
		for i in range(1, len(window_sizes)):
			kernel = np.ones((window_sizes[i], window_sizes[i]), dtype=np.uint8)

			# Apply morphological operations
			filtered_grid = np.asarray(elevation_grid)
			filtered_grid = cv2.erode(filtered_grid, kernel, iterations=1)
			filtered_grid = cv2.dilate(filtered_grid, kernel, iterations=1)

			# Mark points exceeding elevation threshold as non-ground
			flag[np.nonzero(elevation_grid - filtered_grid > elevation_threshold)] = i

			# Compute new elevation threshold
			elevation_threshold = min(
				self.terrain_slope * (window_sizes[i]-window_sizes[i-1]) * self.cell_size + self.initial_threshold, 
				self.max_threshold)

			elevation_grid = filtered_grid


		# Remove interpolated cells & separate ground vs. non-ground points
		keep_cells = np.logical_or(grid_copy[:, :, 0], grid_copy[:, :, 1])
		flag_zeros = (flag == 0)
		flag_nonzeros = np.invert(flag_zeros)

		ground_points = grid_copy[np.logical_and(keep_cells, flag_zeros)]
		non_ground = grid_copy[np.logical_and(keep_cells, flag_nonzeros)]

		return ground_points, non_ground


if __name__ == "__main__":
	las_data = laspy.open(sys.argv[1]).read()
	points = np.asarray(las_data.xyz)

	# Feet to meters
	# points *= 0.3048

	voxelized_points = normals.voxel_filter(points, resolution=2.0)
	preprocessed_points = normals.remove_radius_outliers(voxelized_points)

	# Extract ground surface
	filter = ProgressiveMorphologicalFilter()
	ground, non_ground = filter.extract_ground(preprocessed_points)

	classifications = [2] * ground.shape[0] + [1] * non_ground.shape[0]

	# Write to LiDAR file
	classified_header = laspy.LasHeader(version="1.4", point_format=6)
	classified_las = laspy.LasData(classified_header)
	classified_las.xyz = np.concatenate((ground, non_ground), axis=0)
	classified_las.classification = classifications
	classified_las.write("classified.laz")
