import cv2
import math
import numpy as np
import pyvista as pv
from scipy import interpolate, ndimage
import time
from tqdm import tqdm

from . import utils


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


	def interpolate_holes(self, method='direct',
								interpolation_tile=1000,
								interpolation_padding=100):
		'''
		Fills any holes (i.e. zero values) in grid via interpolation.
		Uses combination of polynomial and nearest-neighbor interpolation.
		'''

		if method not in ['direct', 'incremental']:
			raise Exception('Interpolation method must be one of: "direct", "incremental"')

		# Runs a single interpolate.griddata() across entire grid
		elif method == 'direct':

			nonzeros = np.nonzero(self.point_grid)
			not_nan_or_zero = np.asarray(nonzeros)[:, np.where(~np.isnan(self.point_grid[nonzeros]))[0]]
			zeros = np.where(self.point_grid == 0)

			self.point_grid[zeros[0], zeros[1]] = interpolate.griddata((not_nan_or_zero[0], not_nan_or_zero[1]),
				self.point_grid[(not_nan_or_zero[0], not_nan_or_zero[1])], (zeros[0], zeros[1]));

		# Uses a sliding-window approach to progressively fill in array
		elif method == 'incremental':

			x_padding = utils.divide_ceiling(interpolation_tile - (self.point_grid.shape[0] % interpolation_tile), 2) + interpolation_padding
			y_padding = utils.divide_ceiling(interpolation_tile - (self.point_grid.shape[1] % interpolation_tile), 2) + interpolation_padding

			padded_grid = np.pad(self.point_grid, ((x_padding, x_padding), (y_padding, y_padding)), "symmetric")

			for i in range(interpolation_padding, padded_grid.shape[0]-interpolation_tile, interpolation_tile):
				for j in range(interpolation_padding, padded_grid.shape[1]-interpolation_tile, interpolation_tile):

					interpolation_window = np.asarray(np.meshgrid(np.arange(i - interpolation_padding, i + interpolation_tile + interpolation_padding),
										np.arange(j - interpolation_padding, j + interpolation_tile + interpolation_padding), indexing='ij'))

					interpolation_region = np.asarray(np.meshgrid(np.arange(i, i + interpolation_tile),
										np.arange(j, j + interpolation_tile), indexing='ij'))

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


	def from_point_cloud(point_cloud, resolution, bounds = None):
		'''
		Converts PointCloud object to PointGrid object, using resolution 
		parameter to determine fineness of discretization. (Represents
		X, Y coordinates via grid position; Z coordinate via grid value.)

		If more than one point falls within an XY grid cell, sets cell value
		to be max of points' Z coordinates.
		'''
		if not bounds:
			bounds = point_cloud.bounds

		discretized_grid = np.zeros([int(abs(bounds['max_y'] - bounds['min_y']) / resolution) + 2, 
			int(abs(bounds['max_x'] - bounds['min_x']) / resolution) + 2])

		for i in range(point_cloud.point_cloud.shape[0]):
			x = round((point_cloud.point_cloud[i][0] - bounds['min_x']) / resolution)
			y = round((point_cloud.point_cloud[i][1] - bounds['min_y']) / resolution)
			z = point_cloud.point_cloud[i][2]
			if(z > discretized_grid[y][x]):
				discretized_grid[y][x] = z;

		return PointGrid(point_grid=discretized_grid, resolution=resolution, bounds=bounds);


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