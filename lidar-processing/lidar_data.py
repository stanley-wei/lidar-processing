import cv2
import math
import numpy as np
import pyvista as pv
from scipy import interpolate, ndimage
import time
from tqdm import tqdm

import config
import utils


class PointCloud:

	def __init__(self, point_cloud, bounds = None):
		self.point_cloud = point_cloud
		self._bounds = bounds # Stored as a dictionary; see bounds() function


	def __getitem__(self, index):
		return self.point_cloud[i]


	@property
	def shape(self):
		return self.point_cloud.shape
	

	@property
	def bounds(self):
		if not self._bounds:
			minimums = np.min(self.point_cloud, axis = 0)
			maximums = np.max(self.point_cloud, axis = 0)

			self._bounds = {"max_x": maximums[0], "max_y": maximums[1], "max_z": maximums[2],
				"min_x": minimums[0], "min_y": minimums[1], "min_z": minimums[2]}

		return self._bounds			


	def combine(cloud_1, cloud_2):
		'''
		Combines two PointCloud objects into a new PointCloud containing
		points from both previous PointClouds, updating bounds to match.
		'''
		if not cloud_1:
			return cloud_2
		elif not cloud_2:
			return cloud_1

		return PointCloud(np.concatenate((cloud_1.point_cloud, cloud_2.point_cloud), axis=0),
			bounds=utils.combine_bounds(cloud_1.bounds, cloud_2.bounds))


	def remove_outliers(self, z_mean = None, z_std = None, num_stdev = 2):
		'''
		Removes any outlying points that are [num_stdev] standard deviations above
		the mean. Mean/standard deviation passed as function parameters (if specific 
		values for mean/stdev are desired) or determined from point cloud.
		'''
		mean_z = np.mean(self.point_cloud, axis=0)[2] if z_mean is None else z_mean
		std_z = np.std(self.point_cloud, axis=0)[2] if z_std is None else z_std

		to_delete = np.where(abs(self.point_cloud[2] - mean_z) > num_stdev * std_z)
		self.point_cloud = np.delete(self.point_cloud, to_delete, axis = 0)

		return mean_z, std_z


	def apply_mask(self, mask_image, resolution, start_index=0):
		'''
		Given a mask image, outputs list of all points marked as
		excluded in the mask.
		'''
		excluded_points = []
		for point_i in tqdm(range(self.point_cloud.shape[0])):
			grid_y, grid_x = utils.point_cloud_coord_to_grid(self.point_cloud[point_i][0], 
				self.point_cloud[point_i][1], resolution, bounds=self.bounds)

			if not grid_y >= mask_image.shape[0] and not grid_x >= mask_image.shape[1] and mask_image[grid_y][grid_x] == 0:
				excluded_points.append(point_i + start_index)

		return excluded_points


	def write_to_file(self, file_name, edit_type_ = 'w'):
		file_name = utils.truncate_file_name(file_name)

		file = open(file_name, edit_type_);
		for i in range(self.shape[0]):
			file.write("%f %f %f\n" % (self.point_cloud[i][0], self.point_cloud[i][1], self.point_cloud[i][2]))
	    
		file.close();
		return file_name;


	def to_point_grid(self, resolution, bounds = None):
		'''
		Converts PointCloud object to PointGrid object, using resolution 
		parameter to determine fineness of discretization. (Represents
		X, Y coordinates via grid position; Z coordinate via grid value.)

		If more than one point falls within an XY grid cell, sets cell value
		to be max of points' Z coordinates.
		'''
		if not bounds:
			bounds = self.bounds

		discretized_grid = np.zeros([int(abs(bounds['max_y'] - bounds['min_y']) / resolution) + 2, 
			int(abs(bounds['max_x'] - bounds['min_x']) / resolution) + 2])

		for i in range(self.point_cloud.shape[0]):
			x = round((self.point_cloud[i][0] - bounds['min_x']) / resolution)
			y = round((self.point_cloud[i][1] - bounds['min_y']) / resolution)
			z = self.point_cloud[i][2]
			if(z > discretized_grid[y][x]):
				discretized_grid[y][x] = z;

		return PointGrid(point_grid=discretized_grid, resolution=resolution, bounds=bounds);


	def generate_mesh(self, excluded_points = [], base_height = 0, bounds = []):
		'''
		Uses PyVista library to generate a 3D mesh from PointCloud object,
		using Delaunay triangulation to create a surface and extruding to 
		a plane for volume. Height of extrusion determined by base_height parameter.
		'''
		if not bounds:
			bounds = self.bounds

		pv_cloud = pv.PolyData(self.point_cloud)
		surface = pv_cloud.delaunay_2d()

		if excluded_points:
			surface, indices = surface.remove_points(excluded_points)

		plane = pv.Plane(
			center = (surface.center[0], surface.center[1], pv_cloud.bounds[4] - base_height),
			direction = (0, 0, -1.0),
			i_size = bounds['max_x'] - bounds['min_x'],
			j_size = bounds['max_y'] - bounds['min_y'])

		extruded_mesh = surface.extrude_trim((0, 0, -1.0), plane)
		return extruded_mesh


class PointGrid:

	def __init__(self, point_grid, resolution, bounds):
		self.point_grid = point_grid
		self.resolution = resolution
		self.bounds = bounds


	def __getitem__(self, tuple):
		y, x = tuple
		return self.point_grid[y][x]


	@property
	def shape(self):
		return self.point_grid.shape


	def interpolate_holes(self, method=config.INTERPOLATION_DIRECT):
		'''
		Fills any holes (i.e. zero values) in grid via interpolation.
		Uses combination of polynomial and nearest-neighbor interpolation.
		'''

		# Runs a single interpolate.griddata() across entire grid
		if method == config.INTERPOLATION_DIRECT:

			nonzeros = np.nonzero(self.point_grid)
			not_nan_or_zero = np.asarray(nonzeros)[:, np.where(~np.isnan(self.point_grid[nonzeros]))[0]]
			zeros = np.where(self.point_grid == 0)

			self.point_grid[zeros[0], zeros[1]] = interpolate.griddata((not_nan_or_zero[0], not_nan_or_zero[1]),
				self.point_grid[(not_nan_or_zero[0], not_nan_or_zero[1])], (zeros[0], zeros[1]));

		# Uses a sliding-window approach to progressively fill in array
		elif method == config.INTERPOLATION_INCREMENTAL:

			x_padding = utils.divide_ceiling(config.INTERPOLATION_TILE - (self.point_grid.shape[0] % config.INTERPOLATION_TILE), 2) + config.INTERPOLATION_PADDING
			y_padding = utils.divide_ceiling(config.INTERPOLATION_TILE - (self.point_grid.shape[1] % config.INTERPOLATION_TILE), 2) + config.INTERPOLATION_PADDING

			padded_grid = np.pad(self.point_grid, ((x_padding, x_padding), (y_padding, y_padding)), "symmetric")

			for i in range(config.INTERPOLATION_PADDING, padded_grid.shape[0]-config.INTERPOLATION_TILE, config.INTERPOLATION_TILE):
				for j in range(config.INTERPOLATION_PADDING, padded_grid.shape[1]-config.INTERPOLATION_TILE, config.INTERPOLATION_TILE):

					interpolation_window = np.asarray(np.meshgrid(np.arange(i - config.INTERPOLATION_PADDING, i + config.INTERPOLATION_TILE + config.INTERPOLATION_PADDING),
										np.arange(j - config.INTERPOLATION_PADDING, j + config.INTERPOLATION_TILE + config.INTERPOLATION_PADDING), indexing='ij'))

					interpolation_region = np.asarray(np.meshgrid(np.arange(i, i + config.INTERPOLATION_TILE),
										np.arange(j, j + config.INTERPOLATION_TILE), indexing='ij'))

					nonzeros = np.nonzero(padded_grid[interpolation_window[0], interpolation_window[1]])
					padded_grid[interpolation_region[0], interpolation_region[1]] = interpolate.griddata(nonzeros, padded_grid[nonzeros],
						(interpolation_region[0], interpolation_region[1]))

			point_indices = np.meshgrid(range(x_padding, padded_grid.shape[0] - x_padding),
						range(y_padding, padded_grid.shape[0] - y_padding), indexing='ij')
			self.point_grid = padded_grid[point_indices[0], point_indices[1]]


		# If not able to find values for all zero points via polynomial interpolation,
		# use nearest-neighbors interpolation to fill in remaining zeros.
		holes = np.concatenate((np.where(np.isnan(self.point_grid)), np.where(self.point_grid == 0)), axis=1)
		if holes.any():
			nonzeros = np.nonzero(self.point_grid)
			not_nan_or_zero = np.asarray(nonzeros)[:, np.where(~np.isnan(self.point_grid[nonzeros]))[0]]

			zeros = np.where(self.point_grid == 0)
			self.point_grid[zeros[0], zeros[1]] = interpolate.griddata((not_nan_or_zero[0], not_nan_or_zero[1]),
				self.point_grid[(not_nan_or_zero[0], not_nan_or_zero[1])], 
				(zeros[0], zeros[1]), method='nearest');


	def overlay(base_grid, overlaying_grid):
		'''
		Given two grids of identical size, "stacks" one grid on top of the other.
		Final grid has values of overlaying grid for cells where overlaying grid is nonzero; 
		values of base grid for cells where overlaying grid is zero.
		'''
		overlay_mask = np.zeros([base_grid.shape[0], base_grid.shape[1]], dtype = np.byte);

		# Overlay non-ground points onto PointGrid of ground points
		for i in range(overlaying_grid.shape[0]):
			for j in range(overlaying_grid.shape[1]):
				if overlaying_grid[i, j] != 0:
					overlay_mask[i, j] = 1

		# Fill holes
		overlay_mask = ndimage.binary_fill_holes(overlay_mask).astype(np.byte)

		# Recombine grids
		combined_bounds = utils.combine_bounds(base_grid.bounds, overlaying_grid.bounds)
		combined_grid = PointGrid(point_grid=np.where(overlay_mask == 0, base_grid.point_grid, overlaying_grid.point_grid),
			resolution=base_grid.resolution, bounds=combined_bounds)

		return combined_grid, overlay_mask


	def generate_image(self, file_name):
		'''
		Outputs a grayscale image from PointGrid, using normalized 
		Z values as intensity.
		'''
		max_z = np.max(self.point_grid)
		min_z = np.min(self.point_grid)

		mask_grid = (self.point_grid - min_z) * (255.0 / (max_z - min_z))

		integer_array = np.asarray(mask_grid, dtype=int)
		cv2.imwrite(file_name, integer_array)


	def write_to_file(self, file_name, edit_type_ = 'w'):
		file_name = utils.truncate_file_name(file_name)

		file = open(file_name, edit_type_)
		for i in range(self.shape[0]):
			for j in range(self.shape[1]):
				if self.point_grid[i][j] != 0:
					file.write("%f %f %f\n" % (j * resolution, i * resolution, self.point_grid[i, j]))

		file.close();
		return file_name;


	def to_point_cloud(self, invert_xy=False, ignore_zeros=False):
		'''
		Converts PointGrid object to PointCloud object.
		Parameters:
			- invert_xy: Flips X, Y coordinates of final output
			- ignore_zeros: Do not output points for grid cells with value 0
		'''
		point_cloud = np.zeros((self.point_grid.shape[0] * self.point_grid.shape[1], 3), dtype = int);
		index = 0
		
		i_multi = self.bounds['min_y'] if invert_xy else self.bounds['min_x']
		for i in range(self.point_grid.shape[0]):
			j_multi = self.bounds['min_x'] if invert_xy else self.bounds['min_y']
			for j in range(self.point_grid.shape[1]):
				if not math.isnan(self.point_grid[i, j]) and not (ignore_zeros and self.point_grid[i, j]==0):
					point_cloud[index] = [j_multi, i_multi, 
						self.point_grid[i, j]] if invert_xy else [i_multi, j_multi, self.point_grid[i, j]]
				j_multi += self.resolution
				index += 1
			i_multi += self.resolution

		point_cloud = point_cloud[point_cloud.any(axis=1)]

		return PointCloud(point_cloud, bounds=self.bounds);

